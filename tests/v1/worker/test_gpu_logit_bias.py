# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip(
        "CUDA required for Model Runner V2 logit bias tests",
        allow_module_level=True,
    )

from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.sample.ops.logit_bias import LogitBiasCentering
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.logits_processor import (
    LogitsContext,
    LogitsProcRequestState,
)
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device("cuda")
VOCAB_SIZE = 128
STOP_TOKEN = 3
OTHER_TOKEN = 4
PROMPT_LEN = 2
MIN_TOKENS = 10
# min_len is PROMPT_LEN + MIN_TOKENS, so this position is below the minimum
# and the kernel masks stop tokens.
POS = 5


@pytest.fixture(autouse=True)
def _cpu_default_device():
    # Other sampler suites change the default device; references stay on CPU.
    with torch.device("cpu"):
        yield


def _params(structured: bool) -> SamplingParams:
    return SamplingParams(
        min_tokens=MIN_TOKENS,
        stop_token_ids=[STOP_TOKEN],
        structured_outputs=(
            StructuredOutputsParams(json={"type": "boolean"}) if structured else None
        ),
    )


def _only_stop_token_left(num_rows: int) -> torch.Tensor:
    """Logits as a terminal grammar leaves them: every token masked but one."""
    logits = torch.full((num_rows, VOCAB_SIZE), -float("inf"), device=DEVICE)
    logits[:, STOP_TOKEN] = 1.0
    return logits


def _apply(logits: torch.Tensor, structured: list[bool]) -> torch.Tensor:
    """Run the fused bias kernel with one logits row per request."""
    return _apply_params(logits, [_params(value) for value in structured])


def _apply_params(logits, params, expanded_mapping=None, *, center=False):
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=1,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    # add_request() reads prompt_len per slot; every slot here uses the same one.
    req_states.prompt_len.np[:] = PROMPT_LEN

    state = LogitBiasState(
        None,
        LogitsProcRequestState.from_request_state(req_states),
        _center_logit_bias=center,
    )
    for req_idx, sampling_params in enumerate(params):
        state.add_request(req_idx, sampling_params)
    state.apply_staged_writes()

    num_reqs = len(params)
    num_rows = logits.shape[0]
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    expanded = (
        idx_mapping
        if expanded_mapping is None
        else torch.tensor(expanded_mapping, dtype=torch.int32, device=DEVICE)
    )
    state.apply(
        logits,
        LogitsContext(
            expanded_idx_mapping=expanded,
            idx_mapping=idx_mapping,
            idx_mapping_np=np.arange(num_reqs, dtype=np.intp),
            expanded_local_pos=torch.zeros(num_rows, dtype=torch.int32, device=DEVICE),
            input_ids=torch.zeros(num_rows, dtype=torch.int32, device=DEVICE),
            pos=torch.full((num_rows,), POS, dtype=torch.int32, device=DEVICE),
            seq_lens_upper_bound_np=np.full(num_reqs, POS + 1, dtype=np.int64),
        ),
    )
    return logits.cpu()


@pytest.mark.parametrize("center", [False, True])
def test_v2_logit_bias_exact_offset_with_expanded_mapping_and_allowed_tokens(center):
    """Exercise the kernel mapping, not public bias-plus-speculation support."""
    d = 2.0**-19
    logits = torch.full((3, VOCAB_SIZE), 2.0**30, device=DEVICE)
    logits[0, :2] = torch.tensor([17.0, 17.0 - 2 * d], device=DEVICE)
    logits[2, :2] = logits[0, :2] + 1.0
    original = logits.cpu()
    assert torch.equal(
        original[2, :2].double() - original[0, :2].double(), torch.ones(2)
    )
    params = [
        SamplingParams(),
        SamplingParams(
            logit_bias={0: 15.0 - d, 1: 15.0 + 2 * d}, allowed_token_ids=[0, 1]
        ),
    ]
    out = _apply_params(logits, params, expanded_mapping=[1, 0, 1], center=center)
    assert torch.equal(out[1], original[1])
    assert torch.isneginf(out[[0, 2], 2:]).all()
    assert out[[0, 2]].argmax(dim=-1).tolist() == ([1, 1] if center else [1, 0])


@pytest.mark.parametrize("bias", [None, {}, {0: 0.0}, {0: -0.0}])
@pytest.mark.parametrize("center", [False, True])
def test_v2_logit_bias_unaffected_row_in_mixed_batch(bias, center):
    logits = torch.full((2, VOCAB_SIZE), 17.0, device=DEVICE)
    out = _apply_params(
        logits,
        [SamplingParams(logit_bias=bias), SamplingParams(logit_bias={1: 2.0})],
        center=center,
    )
    assert torch.equal(out[0], torch.full((VOCAB_SIZE,), 17.0))
    assert out[1].argmax().item() == 1


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("center", [False, True])
def test_v2_finite_bias_preserves_min_tokens_empty_row_recovery(structured, center):
    params = _params(structured)
    params.logit_bias = {STOP_TOKEN: 1.0}
    out = _apply_params(_only_stop_token_left(1), [params], center=center)
    assert not torch.isnan(out).any()
    if structured:
        assert torch.isfinite(out[0, STOP_TOKEN])
        assert out.argmax(dim=-1).item() == STOP_TOKEN
    else:
        assert torch.isneginf(out).all()


@pytest.mark.parametrize(
    "values,bias,allowed",
    [
        ([17.0, 17.0 - 2**-18], {0: -15.0, 1: -15.0 + 2**-19}, None),
        ([17.0, 17.0 - 2**-18], {1: 2**-18}, None),
        ([17.0, 16.0, 100.0], {2: 1.0}, [0, 1]),
        ([-torch.inf, -torch.inf], {0: 1.0}, None),
        ([torch.inf, -torch.inf], {0: 1.0}, None),
        ([17.0, torch.inf, torch.nan, -torch.inf], {0: 1.0}, None),
        ([17.0, 16.0], {0: torch.nan}, None),
        ([17.0, 16.0], {0: torch.inf}, None),
        ([17.0, 16.0], {0: -torch.inf}, None),
        ([17.0, 16.0], {0: 2**-150}, None),
        (
            [torch.finfo(torch.float32).max, -torch.finfo(torch.float32).max],
            {0: 1.0},
            None,
        ),
    ],
)
def test_v2_centering_matches_torch_on_edge_cases(values, bias, allowed):
    logits = torch.full((1, VOCAB_SIZE), -torch.inf, device=DEVICE)
    logits[0, : len(values)] = torch.tensor(values, device=DEVICE)
    expected = logits.cpu()
    if allowed:
        mask = torch.ones(VOCAB_SIZE, dtype=torch.bool)
        mask[allowed] = False
        expected[:, mask] = -torch.inf
    LogitBiasCentering({0: bias}, torch.device("cpu")).apply(expected)
    for token, value in bias.items():
        expected[0, token] += value
    actual = _apply_params(
        logits,
        [SamplingParams(logit_bias=bias, allowed_token_ids=allowed)],
        center=True,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    if bias == {1: 2**-18}:
        assert actual.argmax().item() == 0


def test_v2_centering_scans_multiple_blocks_and_respects_row_stride():
    backing = torch.full((3, 8205), 17.0, device=DEVICE)
    logits = backing[:, 3:-3]
    logits[:, -1] = 18.0
    expected = logits.cpu()
    LogitBiasCentering({0: {0: 2.0}, 2: {0: 2.0}}, torch.device("cpu")).apply(expected)
    expected[[0, 2], 0] += 2.0
    actual = _apply_params(
        logits,
        [SamplingParams(logit_bias={0: 2.0}), SamplingParams()],
        expanded_mapping=[0, 1, 0],
        center=True,
    )
    assert torch.equal(actual, expected)
    assert torch.all(backing[:, :3] == 17.0)
    assert torch.all(backing[:, -3:] == 17.0)


def test_v2_min_tokens_restores_stop_token_when_row_would_be_empty():
    """Structured output plus min_tokens must not leave an all -inf row."""
    out = _apply(_only_stop_token_left(1), structured=[True])
    assert out[0, STOP_TOKEN] == 1.0


def test_v2_min_tokens_keeps_empty_row_without_structured_output():
    """Plain min_tokens requests keep the pre-existing masking behavior."""
    out = _apply(_only_stop_token_left(1), structured=[False])
    assert torch.isneginf(out).all()


def test_v2_min_tokens_keeps_masking_when_other_candidates_remain():
    """The fallback must not weaken min_tokens while a non-stop token is legal."""
    logits = _only_stop_token_left(1)
    logits[0, OTHER_TOKEN] = 2.0

    out = _apply(logits, structured=[True])

    assert torch.isneginf(out[0, STOP_TOKEN])
    assert out[0, OTHER_TOKEN] == 2.0


def test_v2_min_tokens_mixed_batch_gates_restore_per_request():
    """CHECK_ALL_MASKED_ROWS is a batch-wide constexpr, so a mixed batch must
    still consult the per-request flag."""
    out = _apply(_only_stop_token_left(2), structured=[True, False])

    assert out[0, STOP_TOKEN] == 1.0
    assert torch.isneginf(out[1]).all()
