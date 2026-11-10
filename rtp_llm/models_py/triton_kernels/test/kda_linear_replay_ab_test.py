"""Correctness and CUDA Graph A/B for the rank-local K3.1 KDA replay path."""

import hashlib
import json
import math
import os
import socket
import statistics
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import triton
from torch import nn

from rtp_llm.models_py.modules.factory.attention.kda_impl import KdaDecode, KdaMetadata
from rtp_llm.models_py.modules.hybrid.kda_attention import KdaAttention
from rtp_llm.models_py.triton_kernels import linear_replay
from rtp_llm.utils.model_weight import W

STEPS = 4
DIM = 128
WIDTH = 4
LOWER_BOUND = -5.0
SAMPLES = 32
ARENAS = 4
LOGS = ("k", "u", "g", "conv_inputs")
HEADERS = ("slot_generations", "log_epochs", "valid_counts", "error_flags")


class KdaProjection:
    """Rank-local FP8 input and forget projections after the all-gather."""

    def __init__(self, case):
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import quant_weight_ue8m0_packed
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
            CudaFp8DeepGEMMLinear,
        )
        from rtp_llm.models_py.triton_kernels.common.activation import (
            quantize_strided_group128_fp8,
        )

        def linear(n, k):
            weight = case.random((n, k), torch.bfloat16) * k**-0.5
            return CudaFp8DeepGEMMLinear(*quant_weight_ue8m0_packed(weight))

        self.width = case.heads * DIM
        self.heads = case.heads
        self.input = linear(case.packed_width, 7168)
        self.forget = linear(self.width, 128)
        self.quantize_forget = quantize_strided_group128_fp8
        self.hidden = self.input.quantize_input(
            case.random((case.batch * STEPS, 7168), torch.bfloat16)
        )

    def __call__(self):
        packed = self.input.forward_quantized(*self.hidden)
        qkv, _, fa, beta = packed[:, : 4 * self.width + 128 + self.heads].split(
            (3 * self.width, self.width, 128, self.heads), dim=-1
        )
        forget = self.forget.forward_quantized(*self.quantize_forget(fa))
        return qkv, forget, beta


class ReplayCase:
    def __init__(self, batch, heads, seed, *, inplace=False, ssm_dtype=torch.float32):
        self.batch = batch
        self.heads = heads
        self.rng = torch.Generator(device="cuda").manual_seed(seed)
        hidden = heads * DIM
        channels = 3 * hidden
        rows = batch * STEPS
        slots = batch + 1
        self.packed_width = triton.cdiv(4 * hidden + 128 + heads, 128) * 128
        self.packed = self.random((rows, self.packed_width), torch.bfloat16)
        self.qkv = self.packed[:, :channels]
        self.beta = self.packed[:, 4 * hidden + 128 : 4 * hidden + 128 + heads]
        self.gate = self.random((rows, hidden), torch.bfloat16)
        config = SimpleNamespace(
            linear_key_head_dim=DIM,
            linear_value_head_dim=DIM,
            linear_num_key_heads=heads,
            linear_num_value_heads=heads,
            linear_conv_kernel_dim=WIDTH,
            ssm_state_dtype=ssm_dtype,
            conv_state_dtype=torch.bfloat16,
        )
        parallelism = SimpleNamespace(tp_size=1)
        weights = {
            W.linear_attn_conv1d_w: self.random((channels, 1, WIDTH), torch.bfloat16),
            W.linear_attn_dt_b_kda: self.random((heads * DIM,)),
            W.linear_attn_alog: self.random((heads,)),
        }
        self.decode = KdaDecode(config, parallelism, weights)
        self.baseline = KdaDecode(config, parallelism, weights)
        self.decode.gate_lower_bound = self.baseline.gate_lower_bound = LOWER_BOUND
        converter = self.decode.linear_cache_converter
        pages = torch.empty(
            (2 * batch + 1, converter.block_size_bytes),
            device="cuda",
            dtype=torch.int8,
        )
        self.ssm = converter.get_ssm_state_tensor(pages)
        self.conv = converter.get_conv_state_tensor(pages)
        self.ssm.uniform_(-0.05, 0.05, generator=self.rng)
        self.conv.uniform_(-0.2, 0.2, generator=self.rng)
        log_shape = (slots, STEPS, heads, DIM)
        self.seed_log = {
            "k": self.random(log_shape),
            "u": self.random(log_shape),
            "g": -self.random(log_shape).abs(),
            "conv_inputs": self.random((slots, STEPS, channels), torch.bfloat16),
        }
        self.cache = SimpleNamespace(
            **{name: tensor.clone() for name, tensor in self.seed_log.items()},
            slot_generations=torch.ones(slots, device="cuda", dtype=torch.int64),
            log_epochs=torch.ones(slots, device="cuda", dtype=torch.int64),
            valid_counts=torch.full((slots,), STEPS, device="cuda", dtype=torch.int32),
            error_flags=torch.zeros(slots, device="cuda", dtype=torch.int32),
        )
        self.layer = SimpleNamespace(
            kv_cache_base=pages,
            linear_replay=self.cache,
            group_id=0,
            seq_size_per_block=64,
        )
        self.baseline._linear_replay = linear_replay.LinearReplay.bind(
            None,
            self.baseline,
            self.layer,
            vector_gate=True,
            state_v_first=False,
            lower_bound=LOWER_BOUND,
        )
        i = torch.arange(batch, device="cuda", dtype=torch.int32)
        self.pages = (i[None, :] + 1, i[None, :] + batch + 1)
        self.inputs = SimpleNamespace(
            slot_ids=i.clone(),
            slot_generations=torch.ones(batch, device="cuda", dtype=torch.int64),
            history_epochs=torch.ones(batch, device="cuda", dtype=torch.int64),
            verify_epochs=torch.full((batch,), 2, device="cuda", dtype=torch.int64),
            history_valid_lengths=torch.full(
                (batch,), STEPS, device="cuda", dtype=torch.int32
            ),
            prev_accept_lengths=i.remainder(STEPS) + 1,
            init_kinds=torch.zeros_like(i),
            state_read_block_ids=self.pages[0].clone(),
            active_block_ids=self.pages[0 if inplace else 1].clone(),
        )
        self.attention_inputs = SimpleNamespace(
            linear_replay=self.inputs, is_prefill=True, valid_token_mask=None
        )
        self.metadata = KdaMetadata(is_target_verify=True)
        self.anchor_seed = (
            (self.ssm[1 : batch + 1].clone(), self.conv[1 : batch + 1].clone())
            if inplace
            else None
        )
        self.projection = None

    def random(self, shape, dtype=torch.float32):
        return torch.empty(shape, dtype=dtype, device="cuda").uniform_(
            -0.25, 0.25, generator=self.rng
        )

    def reset(self):
        for name, src in self.seed_log.items():
            getattr(self.cache, name).copy_(src)
        if self.anchor_seed is not None:
            ssm, conv = self.anchor_seed
            self.ssm[1 : self.batch + 1].copy_(ssm)
            self.conv[1 : self.batch + 1].copy_(conv)

    def advance(self, turn):
        inputs = self.inputs
        inputs.history_epochs.copy_(inputs.verify_epochs)
        inputs.verify_epochs.add_(1)
        inputs.state_read_block_ids.copy_(inputs.active_block_ids)
        if turn % 2 == 0:
            inputs.active_block_ids.copy_(self.pages[0])
        inputs.history_valid_lengths.fill_(STEPS)
        inputs.prev_accept_lengths.copy_(
            (torch.arange(self.batch, device="cuda", dtype=torch.int32) + turn)
            .remainder(STEPS)
            .add_(1)
        )
        inputs.init_kinds.zero_()
        self.packed.uniform_(-0.25, 0.25, generator=self.rng)
        self.gate.uniform_(-0.25, 0.25, generator=self.rng)

    def invoke(self, baseline=False):
        qkv, gate, beta = (
            self.projection() if self.projection else (self.qkv, self.gate, self.beta)
        )
        if baseline:
            qkv = qkv.contiguous()
        decode = self.baseline if baseline else self.decode
        return decode(qkv, gate, beta, self.attention_inputs, self.layer, self.metadata)

    def attention(self):
        # Use the actual packed-layout wrapper and recurrence, with fixed
        # projection results and identity output transforms.
        return KdaAttention(
            input_projection=lambda _: self.packed,
            forget_projection=lambda _: self.gate,
            output_projection=nn.Identity(),
            fa_width=128,
            norm=lambda x, gate, **kwargs: x,
            prefill=self.decode,
            decode=self.decode,
        )

    def finalize(self):
        linear_replay.finalize_linear_replay(self.cache, self.inputs, STEPS)

    def state(self, output):
        return {
            "output": output,
            "ssm": self.ssm,
            "conv": self.conv,
            **{name: getattr(self.cache, name) for name in (*LOGS, *HEADERS)},
        }


def compare(actual, expected):
    for name, ref in expected.items():
        val = actual[name]
        if ref.is_floating_point():
            assert torch.isfinite(val).all().item(), name
        torch.testing.assert_close(val, ref, rtol=0, atol=0, msg=name)


class RecordLaunch:
    def __init__(self, name, launches):
        self.name = name
        self.launches = launches
        self.kernel = getattr(linear_replay, name)

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches.append((self.name, torch.cuda.current_stream().cuda_stream))
            self.kernel[grid](*args, **kwargs)

        return launch


def capture(cases, stream, baseline):
    with torch.cuda.stream(stream):
        for case in cases:
            case.reset()
            case.invoke(baseline)
    stream.synchronize()
    launches = []
    names = ("_rebuild_state_kernel", "_replay_conv_kernel", "_serial_verify_kernel")
    graph = torch.cuda.CUDAGraph()
    with (
        patch.object(linear_replay, names[0], RecordLaunch(names[0], launches)),
        patch.object(linear_replay, names[1], RecordLaunch(names[1], launches)),
        patch.object(linear_replay, names[2], RecordLaunch(names[2], launches)),
        torch.cuda.graph(graph, stream=stream),
    ):
        outputs = [case.invoke(baseline) for case in cases]
    order = names if baseline else (names[1], names[0], names[2])
    assert [name for name, _ in launches] == list(order) * len(cases)
    for i in range(0, len(launches), len(names)):
        streams = dict(launches[i : i + len(names)])
        assert streams[names[1]] == streams[names[2]] == stream.cuda_stream
        assert (streams[names[0]] == stream.cuda_stream) == baseline
    return graph, outputs


def graph_state(captured, cases, stream):
    graph, outputs = captured
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for case in cases:
            case.reset()
            # Detect a missing destination write even when the other arm
            # already wrote the checkpoint.
            case.ssm[case.batch + 1 :].fill_(1.0)
            case.conv[case.batch + 1 :].fill_(1.0)
        graph.replay()
    stream.synchronize()
    return [case.state(out) for out, case in zip(outputs, cases, strict=True)]


def time_graph(graph, cases, stream):
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(SAMPLES)
    ]
    with torch.cuda.stream(stream):
        for _ in range(5):
            for case in cases:
                case.reset()
            graph.replay()
        for start, stop in events:
            for case in cases:
                case.reset()
            start.record()
            graph.replay()
            stop.record()
    stream.synchronize()
    samples = [start.elapsed_time(stop) * 1000 / len(cases) for start, stop in events]
    values = sorted(samples)
    return {
        "p50_us": statistics.median(values),
        "p90_us": values[math.ceil(0.9 * len(values)) - 1],
        "samples_us": samples,
    }


def benchmark(batch, heads, tp, *, project=False, inplace=False):
    cases = [ReplayCase(batch, heads, 101 + i, inplace=inplace) for i in range(ARENAS)]
    if project:
        for case in cases:
            case.projection = KdaProjection(case)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graphs = {
        "baseline": capture(cases, stream, True),
        "optimized": capture(cases, stream, False),
    }
    expected = [
        {key: value.clone() for key, value in state.items()}
        for state in graph_state(graphs["baseline"], cases, stream)
    ]
    for state, ref in zip(
        graph_state(graphs["optimized"], cases, stream), expected, strict=True
    ):
        compare(state, ref)

    trials = []
    for order in (("baseline", "optimized"), ("optimized", "baseline")):
        trial = {name: time_graph(graphs[name][0], cases, stream) for name in order}
        trial["order"] = order
        trials.append(trial)
    result = {
        "batch": batch,
        "queries": STEPS,
        "local_heads": heads,
        "head_dim": DIM,
        "modeled_tp": tp,
        "packed_row_stride": cases[0].packed_width,
        "with_projections": project,
        "inplace_anchor": inplace,
        "trials": trials,
    }
    print(
        "KDA_REPLAY_AB",
        {key: value for key, value in result.items() if key != "trials"},
        [{name: trial[name]["p50_us"] for name in trial["order"]} for trial in trials],
        flush=True,
    )
    return result


class KdaLinearReplayABTest(unittest.TestCase):
    def test_packed_attention_routing(self):
        case = ReplayCase(2, 6, 101)
        expected = case.invoke(baseline=True).flatten(1)
        case.reset()
        seen = []

        def record(_, args):
            seen.append(args[0].stride(0))

        with case.decode.register_forward_pre_hook(record):
            actual = case.attention()(
                case.packed, None, case.layer, case.attention_inputs, case.metadata
            )
        self.assertEqual(seen, [case.packed_width])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        # CUDA paged decode also consumes the packed projection view.
        with (
            patch.object(case.decode, "_conv1d", wraps=lambda qkv, *_: qkv) as conv,
            patch.object(case.decode, "_fla", return_value=None),
        ):
            case.decode(
                case.qkv,
                case.gate,
                case.beta,
                case.attention_inputs,
                case.layer,
                KdaMetadata(),
            )
        self.assertIs(conv.call_args.args[0], case.qkv)

    def test_multiround_graph(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                reference = ReplayCase(5, 6, 83, ssm_dtype=dtype)
                candidate = ReplayCase(5, 6, 83, ssm_dtype=dtype)
                for case in (reference, candidate):
                    case.inputs.slot_ids[-1] = -1
                    case.inputs.init_kinds[:2] = torch.tensor([1, 2], device="cuda")
                    case.inputs.prev_accept_lengths[:2].zero_()
                    case.inputs.history_valid_lengths[:2].zero_()
                    case.inputs.state_read_block_ids[0, 1] = 0

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                graph, (output,) = capture([candidate], stream, False)
                candidate.reset()
                for turn in range(4):
                    if turn:
                        reference.advance(turn)
                        candidate.advance(turn)
                    expected = reference.invoke(baseline=True)
                    graph.replay()
                    reference.finalize()
                    candidate.finalize()
                    compare(candidate.state(output), reference.state(expected))

    def test_graph_replay_ab(self):
        self.assertEqual(torch.cuda.device_count(), 1)
        self.assertEqual(torch.cuda.get_device_capability(), (10, 3))
        props = torch.cuda.get_device_properties(0)
        report = {
            "environment": {
                "host": socket.gethostname(),
                "gpu": props.name,
                "uuid": str(props.uuid),
                "sm": list(torch.cuda.get_device_capability()),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "triton": triton.__version__,
                "replay_sha256": hashlib.sha256(
                    Path(linear_replay.__file__).read_bytes()
                ).hexdigest(),
                "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            "contract": {
                "input": "synthetic packed BF16 QKV, forget and beta; FP32 state/logs",
                "timing": "CUDA-event graph replay through verify/log writes",
                "excluded": "collectives, final header publication, state/log reset, JIT",
                "accept_lengths": "1,2,3,4 repeated",
                "synthetic_gate_lower_bound": LOWER_BOUND,
                "graph_layers": ARENAS,
                "baseline": "KdaDecode with packed QKV copy and serial rebuild/conv",
                "optimized": "KdaDecode with strided QKV and graph rebuild/conv overlap",
                "correctness": "bitwise equality of output, state, logs and headers",
            },
            "shapes": [
                benchmark(8, 24, 4),
                benchmark(40, 24, 4),
                benchmark(40, 24, 4, inplace=True),
                benchmark(160, 6, 16),
                benchmark(40, 24, 4, project=True, inplace=True),
            ],
        }
        path = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]) / "kda_replay_ab.json"
        path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    unittest.main()
