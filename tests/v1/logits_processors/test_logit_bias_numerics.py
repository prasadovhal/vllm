# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU design probes for #61771, not validation of a production hardening fix.

The reference keeps centering outside production until its compatibility policy
is agreed. Run independently of the model fixtures with --confcutdir set to this
directory. Runtime regressions also live in the existing V1/V2 processor suites.
"""

import math
import struct

import pytest
import torch

D = 2.0**-19


@pytest.fixture(autouse=True)
def _use_cpu():
    with torch.device("cpu"):
        yield


def _bias_reference(logits, biases, *, center, allowed=None):
    """Model masking, optional finite centering, then sparse FP32 bias addition."""
    out = logits.clone()
    for row, bias in enumerate(biases):
        if allowed is not None and allowed[row]:
            mask = torch.ones(out.shape[1], dtype=torch.bool)
            mask[allowed[row]] = False
            out[row, mask] = -torch.inf
        bias = {
            token: torch.tensor(value, dtype=torch.float32).item()
            for token, value in (bias or {}).items()
        }
        if center and any(
            math.isfinite(value) and value != 0 and torch.isfinite(out[row, token])
            for token, value in bias.items()
        ):
            finite = torch.isfinite(out[row])
            if finite.any():
                out[row] -= out[row, finite].max()
        for token, value in bias.items():
            out[row, token] += value
    return out


def _exact_shift_pair():
    logits = torch.tensor([[17.0, 17.0 - 2 * D], [18.0, 18.0 - 2 * D]])
    assert logits.dtype == torch.float32
    # Check the stored values in FP64; an FP32 subtraction could hide rounding.
    assert torch.equal(logits[1].double() - logits[0].double(), torch.ones(2))
    return logits


def test_issue_61771_standard_library_reproducer():
    """Post-addition centering cannot undo the rounded tie in the shifted row."""

    def f32(value):
        return struct.unpack("<f", struct.pack("<f", value))[0]

    def center(values):
        return [f32(value - max(values)) for value in values]

    def winner(values):
        return max(range(len(values)), key=values.__getitem__)

    observed = []
    for shift in (0.0, 1.0):
        values = [f32(17.0 + shift), f32(17.0 - 2 * D + shift)]
        bias = [15.0 - D, 15.0 + 2 * D]
        raw = [f32(x + b) for x, b in zip(values, bias)]
        pre = [f32(x + b) for x, b in zip(center(values), bias)]
        observed.append((winner(raw), winner(pre), winner(center(raw))))
    assert observed == [(1, 1, 1), (0, 1, 0)]


@pytest.mark.xfail(
    strict=True,
    reason="#61771: direct FP32 bias addition is not common-offset invariant",
)
def test_direct_fp32_bias_addition_is_offset_invariant():
    logits = _exact_shift_pair()
    biased = logits + torch.tensor([15.0 - D, 15.0 + 2 * D])
    assert biased.argmax(dim=-1).tolist() == [1, 1]


@pytest.mark.parametrize("shift", [1.0, -16.0, -32.0])
@pytest.mark.parametrize(
    "bias", [{0: 15.0 - D, 1: 15.0 + 2 * D}, {0: -15.0, 1: -15.0 + D}]
)
def test_centering_preserves_exact_shift_pairs(shift, bias):
    original = torch.tensor([17.0, 17.0 - 2 * D])
    shifted = original + shift
    assert torch.equal(shifted.double() - original.double(), torch.full((2,), shift))
    out = _bias_reference(torch.stack([original, shifted]), [bias, bias], center=True)
    assert torch.equal(out[0], out[1])
    assert out[0].argmax() == out[1].argmax()


def test_inexact_input_shift_is_outside_invariance_contract():
    original = torch.tensor([1.0, 1.0 + 2.0**-23])
    shifted = original + 1.0
    assert shifted[0] == shifted[1]
    assert not torch.equal(shifted.double() - original.double(), torch.ones(2))
    assert not torch.equal(original - original.max(), shifted - shifted.max())


@pytest.mark.parametrize("bias,winner", [({1: 2 * D}, 0), ({1: 3 * D}, 1)])
def test_centered_near_tie_and_first_index_tie_break(bias, winner):
    out = _bias_reference(_exact_shift_pair(), [bias, bias], center=True)
    assert out.argmax(dim=-1).tolist() == [winner, winner]


def test_mixed_batch_only_centers_rows_with_finite_nonzero_bias():
    biases = [
        None,
        {},
        {0: 0.0},
        {0: -0.0},
        {0: 2.0**-150},
        {0: torch.inf},
        {0: -torch.inf},
        {0: torch.nan},
        {1: 2.0},
    ]
    logits = torch.tensor([[18.0, 17.0]]).repeat(len(biases), 1)
    legacy = _bias_reference(logits, biases, center=False)
    centered = _bias_reference(logits, biases, center=True)
    torch.testing.assert_close(
        centered[:-1], legacy[:-1], rtol=0, atol=0, equal_nan=True
    )
    assert torch.equal(centered[-1], torch.tensor([0.0, 1.0]))
    assert torch.equal(logits, torch.tensor([[18.0, 17.0]]).repeat(len(biases), 1))


@pytest.mark.parametrize(
    "values",
    [
        [-torch.inf, -torch.inf],
        [torch.inf, -torch.inf],
        [torch.nan, -torch.inf],
        [17.0, -torch.inf],
        [17.0, torch.inf],
        [17.0, torch.nan],
    ],
)
def test_centering_does_not_create_nans_or_rewrite_nonfinite_logits(values):
    logits = torch.tensor([values])
    out = _bias_reference(logits, [{0: 1.0}], center=True)
    assert torch.equal(torch.isnan(out), torch.isnan(logits))
    nonfinite = ~torch.isfinite(logits)
    torch.testing.assert_close(out[nonfinite], logits[nonfinite], equal_nan=True)


@pytest.mark.parametrize("all_masked", [False, True])
def test_allowed_tokens_define_the_center_and_empty_rows_stay_masked(all_masked):
    logits = torch.cat([_exact_shift_pair(), torch.full((2, 1), 2.0**30)], dim=1)
    if all_masked:
        logits.fill_(-torch.inf)
    bias = {0: 15.0 - D, 1: 15.0 + 2 * D, 2: 100.0}
    out = _bias_reference(logits, [bias, bias], center=True, allowed=[[0, 1]] * 2)
    assert not torch.isnan(out).any()
    if not all_masked:
        assert out.argmax(dim=-1).tolist() == [1, 1]
        assert torch.equal(out[0, :2], torch.tensor([15.0 - D, 15.0]))
        assert torch.isneginf(out[:, 2]).all()
    else:
        assert torch.isneginf(out).all()


def test_bias_on_only_masked_tokens_does_not_rebase_eligible_scores():
    logits = torch.tensor([[17.0, 16.0, 100.0]])
    bias = [{2: 1.0}]
    legacy = _bias_reference(logits, bias, center=False, allowed=[[0, 1]])
    centered = _bias_reference(logits, bias, center=True, allowed=[[0, 1]])
    assert torch.equal(centered, legacy)


def test_centering_before_repetition_penalty_changes_a_non_tied_winner():
    """Preserving stage order alone does not preserve repetition semantics."""
    logits = torch.tensor([[17.0, 16.0]])
    bias = [{0: 0.25}]
    legacy = _bias_reference(logits, bias, center=False)
    centered = _bias_reference(logits, bias, center=True)
    # Token 0 occurred before; both pipelines apply repetition penalty after bias.
    for scores in (legacy, centered):
        scores[:, 0] = torch.where(scores[:, 0] > 0, scores[:, 0] / 2, scores[:, 0] * 2)
    assert legacy.argmax(dim=-1).item() == 1
    assert centered.argmax(dim=-1).item() == 0
    assert torch.equal(legacy, torch.tensor([[8.625, 16.0]]))
    assert torch.equal(centered, torch.tensor([[0.125, -1.0]]))


def test_raw_scores_are_preserved_but_processed_scores_change():
    raw = _exact_shift_pair()
    saved = raw.clone()
    bias = {0: 15.0 - D, 1: 15.0 + 2 * D}
    legacy = _bias_reference(raw, [bias, bias], center=False)
    centered = _bias_reference(raw, [bias, bias], center=True)
    assert torch.equal(raw, saved)
    assert torch.equal(raw.log_softmax(-1), saved.log_softmax(-1))
    assert not torch.equal(legacy, centered)
    assert not torch.equal(legacy.log_softmax(-1), centered.log_softmax(-1))
    assert torch.equal(centered[0].log_softmax(-1), centered[1].log_softmax(-1))


def test_finite_extremes_can_overflow_during_centering():
    """Finite-only max avoids NaNs but does not guarantee finite differences."""
    limit = torch.finfo(torch.float32).max
    out = _bias_reference(torch.tensor([[limit, -limit]]), [{0: 1.0}], center=True)
    assert not torch.isnan(out).any()
    assert torch.isneginf(out[0, 1])
