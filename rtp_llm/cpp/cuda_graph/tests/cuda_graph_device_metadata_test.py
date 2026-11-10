"""Exercise device-only metadata through the native graph runner, without ATOM.

The oracle uses Python integers, not the model's forward implementation. Captured
host mirrors are poisoned after capture and must remain poisoned after replay.
This catches both a consumer reading host metadata and an unnoticed D2H refresh.
"""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.cpp.cuda_graph.tests import cuda_graph_tagged_cache_test as fixtures
from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner
from rtp_llm.ops.compute_ops import PyModelOutputs

HIDDEN_SIZE = 18
QUERY_WIDTH = 3
HOST_SENTINEL = -113
HOST_FIELDS = (
    "input_lengths",
    "prefix_lengths",
    "sequence_lengths",
    "cu_seqlens",
    "decode_cu_seqlens",
    "kv_cache_kernel_block_id",
    "kv_cache_block_id",
)


def _attention_map(inputs):
    attention = inputs.attention_inputs
    return attention if isinstance(attention, dict) else {"full": attention}


def _cumulative(values):
    result = [0]
    for value in values:
        result.append(result[-1] + value)
    return result


def _device_i32(values):
    return torch.tensor(values, dtype=torch.int32, device="cuda")


class DeviceMetadataModel:
    supports_device_attention_metadata = True
    requires_host_block_table_mirror = False

    def __init__(self, role):
        self.role = role
        self.host_mirrors = {}
        self.forward_calls = 0
        self.probe_calls = 0
        self.prepare_calls = 0

    def cuda_graph_draft_prefill_uses_request_lanes(self):
        return True

    def prepare_fmha_impl(self, inputs, is_cuda_graph=False):
        if not is_cuda_graph:
            raise AssertionError("this regression must exercise native graphs")
        for attention in _attention_map(inputs).values():
            for field in HOST_FIELDS:
                tensor = getattr(attention, field)
                if tensor is not None and tensor.numel() and not tensor.is_cuda:
                    key = (
                        tensor.data_ptr(),
                        tuple(tensor.shape),
                        tuple(tensor.stride()),
                    )
                    self.host_mirrors[key] = tensor
        return self

    def prepare_cuda_graph(self, attention):
        regions = attention.values() if isinstance(attention, dict) else (attention,)
        for region in regions:
            if not region.device_only_metadata:
                raise AssertionError("graph prepare lost the device-only contract")
        self.prepare_calls += 1

    def forward(self, inputs, fmha_impl=None):
        self.forward_calls += 1
        regions = _attention_map(inputs)
        attention = regions["full"]
        if not attention.device_only_metadata:
            # The native runner deliberately keeps its one datatype probe on
            # the legacy dummy-input contract. Real bucket warmups/capture
            # and every replay must use the device-only contract.
            if self.forward_calls != 1:
                raise AssertionError("device metadata disabled after datatype probe")
            self.probe_calls += 1
            return PyModelOutputs(torch.zeros_like(inputs.input_hiddens))
        if not all(region.device_only_metadata for region in regions.values()):
            raise AssertionError("device-only flag differs across cache tags")

        token = torch.arange(inputs.input_ids.numel(), device="cuda")
        if self.role == "decode":
            row = token
        else:
            # Packed draft rows can have varying accepted lengths. Derive the
            # request row from device cumulative lengths, including repeated
            # endpoints for unused request lanes; no host scalar is read.
            row = (token[:, None] >= attention.cu_seqlens_device[None, 1:]).sum(1)
            row = row.clamp(max=attention.input_lengths_device.numel() - 1)
        zero = torch.zeros_like(token, dtype=torch.int32)

        def gather(tensor, offset=0):
            if tensor is None or not tensor.numel():
                return zero
            if not tensor.is_cuda:
                raise AssertionError("captured consumer received host metadata")
            return tensor[row + offset]

        def table_column(tag, physical=False):
            if tag not in regions:
                return zero
            name = (
                "kv_cache_block_id_device"
                if physical
                else "kv_cache_kernel_block_id_device"
            )
            table = getattr(regions[tag], name)
            if table is None or not table.numel():
                return zero
            weights = torch.arange(
                1, table.shape[1] + 1, device="cuda", dtype=torch.int32
            )
            return (table[row] * weights).sum(1)

        table_sum = sum(
            region.kv_cache_kernel_block_id_device.sum() for region in regions.values()
        )
        with_decode_metadata = self.role in ("decode", "target_verify")
        columns = (
            gather(attention.input_lengths_device),
            gather(attention.prefix_lengths_device),
            gather(attention.cu_seqlens_device),
            gather(attention.cu_seqlens_device, 1),
            gather(attention.cu_kv_seqlens_device),
            gather(attention.cu_kv_seqlens_device, 1),
            (
                gather(attention.sequence_lengths_plus_1_device)
                if with_decode_metadata
                else zero
            ),
            (
                gather(attention.decode_cu_seqlens_device, 1)
                if with_decode_metadata
                else zero
            ),
            attention.padding_offset[token],
            table_column("full"),
            table_column("full", physical=True),
            table_column("aux"),
            table_column("aux", physical=True),
            inputs.input_ids,
            inputs.combo_position_ids,
            table_sum.expand_as(token),
            (
                attention.cu_kv_seqlens_device[-1].expand_as(token)
                if self.role != "decode"
                else zero
            ),
            (
                attention.sequence_lengths_plus_1_device.sum().expand_as(token)
                if with_decode_metadata
                else zero
            ),
        )
        return PyModelOutputs(torch.stack(columns, 1).to(inputs.input_hiddens.dtype))


class DeviceMetadataGraphTest(unittest.TestCase):
    def test_target_verify_device_metadata_survives_batch_shrink_and_growth(self):
        self._check_role("target_verify", ([3] * 4, [3], [3] * 3, [3] * 4))

    def test_decode_device_metadata_survives_batch_shrink_and_growth(self):
        self._check_role("decode", ([1] * 4, [1], [1] * 3, [1] * 4))

    def test_draft_prefill_device_metadata_survives_bucket_reuse(self):
        # Same request capacity with different packed lengths reuses a bucket;
        # smaller batches then return to the former larger bucket as well.
        self._check_role(
            "draft_prefill",
            ([3] * 4, [1, 3, 2, 1], [1], [3, 1, 2], [3] * 4, [2, 1, 3, 2]),
        )

    @torch.inference_mode()
    def _check_role(self, role, shapes):
        torch.cuda.set_device(0)
        model = DeviceMetadataModel(role)
        runner = CudaGraphRunner()
        with patch.dict(os.environ, {"RTP_LLM_DEVICE_ATTENTION_METADATA": "1"}):
            if role == "draft_prefill":
                runner.init_mtp_draft_prefill(model, 4, HIDDEN_SIZE, QUERY_WIDTH, 1)
            else:
                runner.init_decode(
                    model,
                    HIDDEN_SIZE,
                    64,
                    2,
                    [4],
                    ["full", "aux"],
                    role == "target_verify",
                    QUERY_WIDTH if role == "target_verify" else 1,
                    1,
                )
        self.assertEqual(model.probe_calls, 1)
        self.assertTrue(model.host_mirrors)
        captured_forward_calls = model.forward_calls
        # Capture has completed and synchronized. Poison the real captured
        # host buffers, not detached clones or only the request-side metadata.
        for tensor in model.host_mirrors.values():
            tensor.fill_(HOST_SENTINEL)

        buckets = []
        for iteration, lengths in enumerate(shapes):
            with self.subTest(role=role, iteration=iteration, lengths=lengths):
                inputs, expected, source_host = self._make_case(
                    role, list(lengths), iteration
                )
                self.assertTrue(runner.canRun(inputs))
                buckets.append(runner.getCurrentRealGraphSize())
                prepare_before = model.prepare_calls
                # Exercise the same metadata-only prepare followed by forward
                # used by the executor's asynchronous prepare worker.
                self.assertTrue(runner.prepare(inputs, False))
                self.assertGreater(model.prepare_calls, prepare_before)
                actual = runner.forward(inputs).hidden_states.clone()
                torch.cuda.synchronize()
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
                self.assertEqual(model.forward_calls, captured_forward_calls)
                for tensor in model.host_mirrors.values():
                    self.assertTrue(torch.all(tensor == HOST_SENTINEL).item())
                for tensor, snapshot in source_host:
                    torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)
        if role == "draft_prefill":
            self.assertEqual(buckets[0], buckets[1])
            self.assertEqual(buckets[0], buckets[-1])
        else:
            self.assertEqual(buckets, [4] * len(shapes))

    def _make_case(self, role, lengths, iteration):
        batch = len(lengths)
        tags = ["full"] if role == "draft_prefill" else ["full", "aux"]
        values = {tag: 1 for tag in tags}
        if role == "decode":
            inputs = fixtures._build_decode_inputs(tags, values, batch, 2)
        elif role == "target_verify":
            inputs = fixtures._build_target_verify_inputs(
                tags, values, batch, QUERY_WIDTH, 1
            )
        else:
            inputs = fixtures._build_prefill_inputs(tags, values, lengths)
            inputs.attention_inputs = inputs.attention_inputs["full"]

        prefix = (
            [] if role == "decode" else [iteration + 1 + row for row in range(batch)]
        )
        sequence = [iteration + 4 + row for row in range(batch)]
        original_lengths = (
            [iteration + 2 + row for row in range(batch)]
            if role == "decode"
            else lengths
        )
        cu_q = _cumulative(lengths)
        cu_kv = _cumulative(
            [
                length + (prefix[row] if prefix else sequence[row])
                for row, length in enumerate(lengths)
            ]
        )
        seq_plus_one = [
            value + 1 for value in (sequence if role == "decode" else prefix)
        ]
        token_count = sum(lengths)
        token_ids = [iteration + token for token in range(token_count)]
        positions = [
            (prefix[row] if prefix else sequence[row]) + offset
            for row, length in enumerate(lengths)
            for offset in range(length)
        ]
        stride = QUERY_WIDTH + 1
        padding = [
            row * stride - cu_q[row]
            for row, length in enumerate(lengths)
            for _ in range(length)
        ]
        width = 1 if role == "draft_prefill" else 1 + iteration % 2
        tables = {}
        source_host = []
        for tag_index, (tag, attention) in enumerate(
            sorted(_attention_map(inputs).items())
        ):
            attention.device_only_metadata = True
            # The test wrapper publishes readonly *_device fields from these
            # raw length tensors, matching PyWrappedModel's device inputs.
            attention.input_lengths = _device_i32(original_lengths)
            attention.prefix_lengths = _device_i32(prefix)
            attention.sequence_lengths = _device_i32(
                sequence if role == "decode" else []
            )
            attention.cu_seqlens_device = _device_i32(cu_q)
            attention.cu_kv_seqlens_device = _device_i32(cu_kv)
            attention.sequence_lengths_plus_1_device = _device_i32(seq_plus_one)
            attention.decode_cu_seqlens_device = _device_i32(list(range(batch + 1)))
            attention.padding_offset = _device_i32(padding)
            # A host capacity bound is deliberately not the exact GPU total.
            # Padded target rows must extend the device endpoint, not this bound.
            attention.context_total_kv_length = cu_kv[-1] + 37
            kernel = [
                [iteration + tag_index + row + col + 1 for col in range(width)]
                for row in range(batch)
            ]
            physical = [[value + 7 for value in row] for row in kernel]
            tables[tag] = (kernel, physical)
            attention.kv_cache_kernel_block_id_device = _device_i32(kernel)
            attention.kv_cache_block_id_device = _device_i32(physical)
            for field, shape in (
                ("cu_seqlens", (batch + 1,)),
                ("decode_cu_seqlens", (batch + 1,)),
                ("kv_cache_kernel_block_id", (batch, width)),
                ("kv_cache_block_id", (batch, width)),
            ):
                tensor = torch.full(
                    shape, HOST_SENTINEL - iteration, dtype=torch.int32
                ).pin_memory()
                setattr(attention, field, tensor)
                source_host.append((tensor, tensor.clone()))

        # Rebind the tag map to refresh its C++ anchor after replacing fields.
        inputs.attention_inputs = inputs.attention_inputs
        inputs.input_ids = _device_i32(token_ids)
        inputs.combo_position_ids = _device_i32(positions)
        inputs.input_hiddens = torch.zeros(
            (token_count, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda"
        )

        def table_value(tag, row, physical=False):
            if tag not in tables or (physical and role == "draft_prefill"):
                return 0
            return sum(
                (col + 1) * value
                for col, value in enumerate(tables[tag][int(physical)][row])
            )

        table_sum = sum(
            value for kernel, _ in tables.values() for row in kernel for value in row
        )
        padded_requests = 0 if role == "draft_prefill" else 4 - batch
        padded_kv_end = (
            cu_kv[-1] + padded_requests * QUERY_WIDTH
            if role == "target_verify"
            else cu_kv[-1]
        )
        sequence_sum = sum(seq_plus_one) + padded_requests * (
            QUERY_WIDTH if role == "target_verify" else 1
        )
        expected = []
        for row, length in enumerate(lengths):
            for offset in range(length):
                token = cu_q[row] + offset
                expected.append(
                    [
                        original_lengths[row],
                        prefix[row] if prefix else 0,
                        cu_q[row],
                        cu_q[row + 1],
                        cu_kv[row],
                        cu_kv[row + 1],
                        seq_plus_one[row] if role != "draft_prefill" else 0,
                        row + 1 if role != "draft_prefill" else 0,
                        padding[token],
                        table_value("full", row),
                        table_value("full", row, True),
                        table_value("aux", row),
                        table_value("aux", row, True),
                        token_ids[token],
                        positions[token],
                        table_sum,
                        padded_kv_end if role != "decode" else 0,
                        sequence_sum if role != "draft_prefill" else 0,
                    ]
                )
        return inputs, torch.tensor(expected, dtype=torch.bfloat16), source_host


if __name__ == "__main__":
    unittest.main()
