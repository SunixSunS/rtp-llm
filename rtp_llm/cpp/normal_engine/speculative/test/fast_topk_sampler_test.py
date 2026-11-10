"""Exercise the native tail sampler, its route guards, and output ownership."""

import importlib
import unittest
from unittest.mock import patch

import torch

native = importlib.import_module(
    "rtp_llm.cpp.normal_engine.speculative.test.libtest_fast_topk_sampler"
)
selector = importlib.import_module("aiter.ops.topk_select")


class FastTopKSamplerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Do not prime AITER: a width-one first call must use the legacy route
        # without specializing the fast selector for subsequent wider tensors.
        scores = torch.ones((16, 1), device="cuda")
        with patch.object(selector, "topk_select", wraps=selector.topk_select) as spy:
            probs, ids = native.sample(scores)
        if spy.call_count:
            raise AssertionError("width-one input entered AITER")
        torch.testing.assert_close(ids, torch.zeros_like(ids), rtol=0, atol=0)
        torch.testing.assert_close(probs, scores, rtol=0, atol=0)

    def check_sampler(self, scores, expect_fast, mapping=None):
        expected = scores.argmax(dim=-1, keepdim=True)
        expected_probs = torch.zeros_like(scores).scatter_(-1, expected, 1)
        expected_ids = expected if mapping is None else mapping[expected]
        with patch.object(selector, "topk_select", wraps=selector.topk_select) as spy:
            probs, ids = native.sample(scores, mapping, True)
        self.assertEqual(spy.call_count, int(expect_fast))
        legacy_probs, legacy_ids = native.sample(scores, mapping, False)
        torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(ids, legacy_ids, rtol=0, atol=0)
        torch.testing.assert_close(probs, expected_probs, rtol=0, atol=0)
        torch.testing.assert_close(probs, legacy_probs, rtol=0, atol=0)
        saved_probs = probs.clone()
        saved_ids = ids.clone()
        native.sample(scores.neg(), mapping, True)
        torch.testing.assert_close(probs, saved_probs, rtol=0, atol=0)
        torch.testing.assert_close(ids, saved_ids, rtol=0, atol=0)

    def test_width_dtype_and_layout_guards_on_both_streams(self):
        cases = (
            (16, 13, "dense", False),
            (16, 16384, "dense", False),
            (16, 16385, "dense", True),
            (16, 248320, "dense", True),
            (1, 16385, "singleton_stride", False),
            (16, 16385, "strided", False),
            (16, 16385, "bf16", False),
        )
        for stream in (torch.cuda.default_stream(), torch.cuda.Stream()):
            with torch.cuda.stream(stream):
                for batch, width, layout, expect_fast in cases:
                    with self.subTest(batch=batch, width=width, layout=layout):
                        scores = torch.full((batch, width), -20.0, device="cuda")
                        # Distinct maxima and low-index ties across rows.
                        rows = torch.arange(batch, device="cuda")
                        scores[rows, rows.remainder(11)] = 4.0
                        scores[:, -1] = 4.0
                        if layout == "singleton_stride":
                            scores = scores.as_strided((1, width), (1, 1))
                        elif layout == "strided":
                            backing = torch.empty((batch, width * 2), device="cuda")
                            scores = backing[:, ::2].copy_(scores)
                        elif layout == "bf16":
                            scores = scores.to(torch.bfloat16)
                        mapping = torch.arange(width, device="cuda") * 3 + 7
                        self.check_sampler(scores, expect_fast)
                        self.check_sampler(scores, expect_fast, mapping)

    def test_batch_guard_boundary(self):
        for batch in (1023, 1024):
            with self.subTest(batch=batch):
                scores = torch.zeros((batch, 16385), device="cuda")
                scores[:, -1] = 1.0
                self.check_sampler(scores, batch < 1024)

    def test_nonfinite_values_preserve_legacy_selection(self):
        scores = torch.zeros((8, 16385), device="cuda")
        scores[0].fill_(float("-inf"))
        scores[1].fill_(float("inf"))
        scores[2].fill_(float("nan"))
        scores[3, 7] = scores[3, 13] = float("nan")
        scores[4, 7] = scores[4, 13] = float("inf")
        scores[5, -1] = float("nan")
        scores[6].fill_(-0.0)
        scores[7, -1] = 1.0
        self.check_sampler(scores, True)


if __name__ == "__main__":
    unittest.main()
