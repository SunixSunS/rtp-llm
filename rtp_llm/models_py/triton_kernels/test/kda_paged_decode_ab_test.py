"""CUDA Graph A/B for K3.1 TP4 KDA with Linear Replay disabled."""

import hashlib
import json
import math
import os
import socket
import statistics
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import triton
from torch import nn

from rtp_llm.models_py.modules.factory.attention import kda_impl
from rtp_llm.models_py.modules.hybrid.kda_attention import KdaAttention
from rtp_llm.models_py.triton_kernels.common import activation
from rtp_llm.models_py.triton_kernels.kda import fused_recurrent as recurrent
from rtp_llm.utils.model_weight import W

TP = 4
DP = 4
HEADS = 24
DIM = 128
WIDTH = HEADS * DIM
CHANNELS = 3 * WIDTH
CONV_WIDTH = 4
PACKED_STRIDE = triton.cdiv(4 * WIDTH + 128 + HEADS, 128) * 128
BLOCK_TOKENS = 128
ARENAS = 4
SAMPLES = 32


class DecodeCase:
    def __init__(self, batch, queries, seed, *, inactive=False):
        self.batch, self.queries = batch, queries
        self.generator = torch.Generator(device="cuda").manual_seed(seed)
        self.packed = self.random((batch * queries, PACKED_STRIDE), torch.bfloat16)
        self.qkv = self.packed[:, :CHANNELS]
        self.beta = self.packed[:, 4 * WIDTH + 128 : 4 * WIDTH + 128 + HEADS]
        self.gate = self.random((batch * queries, WIDTH), torch.bfloat16)
        self.output_gate = self.packed[:, CHANNELS : 4 * WIDTH].view(-1, HEADS, DIM)
        self.norm_weight = self.random((DIM,), torch.bfloat16) + 1
        self.valid_rows = torch.arange(batch * queries, device="cuda").remainder(7) != 0
        if inactive:
            self.valid_rows[-queries:] = False

        config = SimpleNamespace(
            linear_key_head_dim=DIM,
            linear_value_head_dim=DIM,
            linear_num_key_heads=TP * HEADS,
            linear_num_value_heads=TP * HEADS,
            linear_conv_kernel_dim=CONV_WIDTH,
            ssm_state_dtype=torch.float32,
            conv_state_dtype=torch.bfloat16,
        )
        self.decode = kda_impl.KdaDecode(
            config,
            SimpleNamespace(tp_size=TP),
            {
                W.linear_attn_conv1d_w: self.random(
                    (CHANNELS, 1, CONV_WIDTH), torch.bfloat16
                ),
                W.linear_attn_dt_b_kda: self.random((WIDTH,), torch.float32),
                W.linear_attn_alog: self.random((HEADS,), torch.float32),
            },
        )
        self.decode.gate_lower_bound = -5.0
        converter = self.decode.linear_cache_converter
        blocks = batch * (queries + 1) + 1
        pages = torch.zeros(
            (blocks, converter.block_size_bytes), device="cuda", dtype=torch.int8
        )
        self.ssm = converter.get_ssm_state_tensor(pages)
        self.conv = converter.get_conv_state_tensor(pages)
        self.layer = SimpleNamespace(
            kv_cache_base=pages,
            seq_size_per_block=BLOCK_TOKENS,
        )
        # Every query has a separate checkpoint. Alternate source/destination
        # aliasing, and use different physical pages per request.
        ids = torch.randperm(blocks - 1, device="cuda", generator=self.generator)
        ids = (ids + 1).reshape(batch, queries + 1)
        source = ids[:, 0].contiguous()
        destinations = ids[:, 1:].clone()
        destinations[1::2, 0] = source[1::2]
        table = torch.full((batch, queries + 2), -1, device="cuda", dtype=torch.int32)
        table[::2, 1] = source[::2]
        table[:, 2:] = destinations
        prefix = torch.full((batch,), 2 * BLOCK_TOKENS, device="cuda", dtype=torch.int32)
        prefix[1::2] += 7
        if inactive:
            # A positive length with a null source makes recurrence skip the row.
            table[-1, :].fill_(-1)
            table[-1, 1:3] = 0
            prefix[-1] = 2 * BLOCK_TOKENS
        self.inputs = SimpleNamespace(
            linear_replay=None,
            prefix_lengths=prefix,
            kv_cache_kernel_block_id_device=table,
            sequence_lengths_plus_1_device=prefix + 1,
        )
        self.metadata = kda_impl.KdaMetadata(is_target_verify=queries > 1)
        self.source = source
        self.ssm_seed = self.random((batch, HEADS, DIM, DIM), torch.float32)
        self.conv_seed = self.random((batch, CONV_WIDTH - 1, CHANNELS), torch.bfloat16)

    def random(self, shape, dtype):
        return torch.empty(shape, dtype=dtype, device="cuda").uniform_(
            -0.125, 0.125, generator=self.generator
        )

    def reset(self, *, clear=False):
        if clear:
            self.layer.kv_cache_base.zero_()
        self.ssm.index_copy_(0, self.source, self.ssm_seed)
        self.conv.index_copy_(0, self.source, self.conv_seed)

    def invoke(self, optimized):
        if optimized:
            return self.decode(
                self.qkv, self.gate, self.beta, self.inputs, self.layer, self.metadata
            )

        pages = self.layer.kv_cache_base
        convolved = self.decode._conv1d(
            self.qkv.contiguous(),
            pages,
            BLOCK_TOKENS,
            self.inputs,
            self.metadata.is_target_verify,
        )
        return self.decode._fla(
            convolved,
            self.gate,
            self.beta,
            pages,
            BLOCK_TOKENS,
            self.inputs,
            self.metadata.is_target_verify,
        )

    def state(self, output):
        result = {"ssm": self.ssm, "conv": self.conv}
        if isinstance(output, tuple):
            values, scales = output
            result.update(output=values.view(torch.int8), scales=scales)
        else:
            result["output"] = output
        return result

    def invoke_norm(self, optimized):
        output = self.invoke(optimized)
        gate = self.output_gate
        mask = self.valid_rows
        if not optimized:
            output = torch.where(mask[:, None, None], output, 0)
            gate = torch.where(mask[:, None, None], gate, 0)
            mask = None
        return activation.rmsnorm_sigmoid_gate_per_token_group_quant_fp8(
            output, gate, self.norm_weight, 1e-6, valid_rows=mask
        )


class RecurrentLaunch:
    def __init__(self, optimized):
        self.kernel = recurrent.fused_recurrent_kda_fwd_kernel
        self.optimized = optimized
        self.count = 0

    def __getitem__(self, grid):
        def launch(**kwargs):
            self.count += 1
            assert kwargs["block_map"] is not None and kwargs["BV"] == 16
            assert kwargs["num_warps"] == 1
            assert kwargs["ZERO_INVALID"] == self.optimized
            expected = CHANNELS if self.optimized else WIDTH
            assert all(
                kwargs[f"stride_{key}_token"] == expected for key in ("q", "k", "v")
            )
            self.kernel[grid](**kwargs)

        return launch


def capture(cases, stream, optimized, *, norm=False):
    stream.wait_stream(torch.cuda.current_stream())
    recurrent_launch = RecurrentLaunch(optimized)
    conv_kernel = kda_impl.causal_conv1d_update
    conv_count = 0
    fwd = partial(
        recurrent.fused_recurrent_kda_fwd, zero_invalid_in_kernel=optimized
    )

    def invoke(case):
        return case.invoke_norm(optimized) if norm else case.invoke(optimized)

    def conv(x, *args, **kwargs):
        nonlocal conv_count
        conv_count += 1
        # [B, C, Q] is a view with channels contiguous. Use the batch stride
        # for Q=1 because its singleton token axis has no meaningful pitch.
        pitch = x.stride(0) // x.shape[2]
        assert pitch == (PACKED_STRIDE if optimized else CHANNELS)
        return conv_kernel(x, *args, **kwargs)

    with (
        patch.object(recurrent, "fused_recurrent_kda_fwd_kernel", recurrent_launch),
        patch.object(recurrent, "fused_recurrent_kda_fwd", fwd),
        patch.object(kda_impl, "causal_conv1d_update", conv),
    ):
        with torch.cuda.stream(stream):
            for case in cases:
                case.reset()
                invoke(case)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = [invoke(case) for case in cases]
    assert conv_count == recurrent_launch.count == 2 * len(cases)
    return graph, outputs


def read_graph(graph_and_outputs, cases, stream):
    graph, outputs = graph_and_outputs
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for case in cases:
            case.reset(clear=True)
        graph.replay()
    stream.synchronize()
    return [case.state(out) for case, out in zip(cases, outputs, strict=True)]


def compare(actual, expected):
    for name, ref in expected.items():
        value = actual[name]
        torch.testing.assert_close(value, ref, rtol=0, atol=0, msg=name)
        assert torch.isfinite(value).all().item(), name


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
    return {
        "p50_us": statistics.median(samples),
        "p90_us": sorted(samples)[math.ceil(0.9 * len(samples)) - 1],
        "samples_us": samples,
    }


def benchmark(batch, queries, *, inactive=False, norm=False):
    cases = [
        DecodeCase(batch, queries, 101 + i, inactive=inactive)
        for i in range(1 if inactive else ARENAS)
    ]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    base = capture(cases, stream, False, norm=norm)
    expected = [
        {key: value.clone() for key, value in state.items()}
        for state in read_graph(base, cases, stream)
    ]
    candidate = capture(cases, stream, True, norm=norm)
    actual = read_graph(candidate, cases, stream)
    for state, reference in zip(actual, expected, strict=True):
        compare(state, reference)
    result = {
        "batch": batch, "queries": queries, "inactive": inactive, "with_output_norm": norm
    }
    if not inactive:
        graphs = {"baseline": base[0], "optimized": candidate[0]}
        result["trials"] = [
            {
                "order": order,
                **{name: time_graph(graphs[name], cases, stream) for name in order},
            }
            for order in (("baseline", "optimized"), ("optimized", "baseline"))
        ]
    return result


class KdaPagedDecodeABTest(unittest.TestCase):
    def test_zero_length_padding(self):
        reference = DecodeCase(4, 4, 73, inactive=True)
        optimized = DecodeCase(4, 4, 73, inactive=True)
        reference.reset(clear=True)
        optimized.reset(clear=True)
        optimized.inputs.sequence_lengths_plus_1_device[-1] = 0
        optimized.inputs.kv_cache_kernel_block_id_device[-1].fill_(-1)
        expected = reference.invoke(False)
        actual = optimized.invoke(True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Page zero is scratch for the baseline convolution. Every live page
        # and its checkpoint must be identical when the padded row is skipped.
        torch.testing.assert_close(optimized.ssm[1:], reference.ssm[1:], rtol=0, atol=0)
        torch.testing.assert_close(optimized.conv[1:], reference.conv[1:], rtol=0, atol=0)

    def test_masked_attention_graph(self):
        class QuantizedOutput(nn.Module):
            def forward_quantized(self, values, scales):
                self.scales = scales
                return values

        case = DecodeCase(4, 4, 109, inactive=True)
        mask_storage = torch.empty(2 * case.valid_rows.numel(), device="cuda", dtype=torch.bool)
        mask = mask_storage[::2]
        mask.copy_(case.valid_rows)
        case.valid_rows = mask
        case.inputs.is_prefill = True
        case.inputs.valid_token_mask = mask
        prefill = SimpleNamespace(
            parallelism_config=SimpleNamespace(tp_size=1),
            local_num_k_heads=HEADS,
            local_num_v_heads=HEADS,
            head_k_dim=DIM,
            head_v_dim=DIM,
        )
        projection = QuantizedOutput()
        attention = KdaAttention(
            input_projection=lambda _: case.packed,
            forget_projection=lambda _: case.gate,
            output_projection=projection,
            fa_width=128,
            norm=SimpleNamespace(weight=case.norm_weight, eps=1e-6),
            prefill=prefill,
            decode=case.decode,
            fp8_output_norm=True,
        )

        def invoke():
            return attention(case.packed, None, case.layer, case.inputs, case.metadata)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            case.reset(clear=True)
            invoke()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = invoke()
        scales = projection.scales
        for pattern in (0, 1, 2):
            mask.copy_(
                torch.arange(mask.numel(), device="cuda").remainder(3) == pattern
            )
            if pattern == 1:
                mask.fill_(True)
            if pattern == 2:
                mask.fill_(False)
            case.reset(clear=True)
            expected, expected_scales = case.invoke_norm(False)
            expected = expected.view(torch.int8).clone()
            expected_scales = expected_scales.clone()
            case.reset(clear=True)
            graph.replay()
            torch.testing.assert_close(actual.view(torch.int8), expected, rtol=0, atol=0)
            torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)

        # Masking must suppress unspecified core values before RMSNorm.
        values = torch.full((16, HEADS, DIM), float("nan"), device="cuda", dtype=torch.bfloat16)
        gates = torch.full_like(values, float("nan"))
        expected = activation.rmsnorm_sigmoid_gate_per_token_group_quant_fp8(
            torch.zeros_like(values), torch.zeros_like(gates), case.norm_weight, 1e-6
        )
        actual = activation.rmsnorm_sigmoid_gate_per_token_group_quant_fp8(
            values, gates, case.norm_weight, 1e-6, valid_rows=mask
        )
        torch.testing.assert_close(actual[0].view(torch.int8), expected[0].view(torch.int8), rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)

    def test_graph_ab(self):
        self.assertEqual(torch.cuda.device_count(), 1)
        self.assertEqual(torch.cuda.get_device_capability(), (10, 3))
        device = torch.cuda.get_device_properties(0)
        report = {
            "environment": {
                "host": socket.gethostname(),
                "gpu": device.name,
                "uuid": str(device.uuid),
                "sm": list(torch.cuda.get_device_capability()),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "triton": triton.__version__,
                "source_sha256": {
                    Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                    for path in (__file__, kda_impl.__file__, recurrent.__file__, activation.__file__)
                },
            },
            "contract": {
                "input": "synthetic BF16 packed K3.1 projection; BF16 conv, FP32 SSM",
                "modeled_tp": TP,
                "modeled_dp": DP,
                "local_heads": HEADS,
                "head_dim": DIM,
                "packed_row_stride": PACKED_STRIDE,
                "linear_replay": False,
                "page_tokens": BLOCK_TOKENS,
                "page_mapping": "one checkpoint per query, mixed source/destination aliasing",
                "baseline": "paged KDA with four QKV copies and output pre-zeroing",
                "optimized": "KdaDecode.forward with strided QKV and in-kernel empty-row initialization",
                "timing": "single-rank CUDA Graph conv through recurrent and checkpoint writes",
                "excluded": "projections, collectives, JIT, cache reset, graph capture",
                "graph_layers": ARENAS,
            },
            "correctness": benchmark(4, 4, inactive=True),
            "masked_correctness": benchmark(4, 4, inactive=True, norm=True),
            "shapes": [benchmark(b, q) for b, q in ((8, 4), (40, 4), (80, 4), (40, 1))],
            "masked_shapes": [benchmark(40, 4, norm=True)],
        }
        path = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]) / "kda_paged_decode_ab.json"
        path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    unittest.main()
