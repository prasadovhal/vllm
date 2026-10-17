# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip(
        "CUDA required for Model Runner V2 logits processor tests",
        allow_module_level=True,
    )

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.logits_processor import (
    LogitsContext,
    LogitsProcessor,
    LogitsProcRequestState,
)
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device("cuda")
VOCAB_SIZE = 128
TARGET_TOKEN = 7
BIAS = 5.0
BOOST = 100.0
TEMPERATURE = 2.0


@pytest.fixture(autouse=True)
def _cpu_default_device():
    with torch.device("cpu"):
        yield


class MockReasoningConfig:
    reasoning_start_token_ids = [90]
    reasoning_end_token_ids = [91]
    natural_reasoning_end_token_ids = [91]


class RecordingProcessor(LogitsProcessor):
    """Adds BOOST to TARGET_TOKEN and records every lifecycle callback."""

    def __init__(self, vllm_config: Any, req_state: Any):
        self.events: list[str] = []
        self.seen_logits: torch.Tensor | None = None

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        self.events.append(f"add_request:{req_idx}")
        return True

    def apply_staged_writes(self) -> None:
        self.events.append("apply_staged_writes")

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        self.events.append("apply")
        self.seen_logits = logits.clone()
        logits[:, TARGET_TOKEN] += BOOST
        return logits


def _make_sampler(processor: RecordingProcessor | None = None) -> Sampler:
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=1,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    return Sampler(
        vllm_config=SimpleNamespace(reasoning_config=MockReasoningConfig()),
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
        req_states=req_states,
        custom_logits_processors=[processor] if processor is not None else [],
    )


def test_custom_processor_lifecycle_and_pipeline_position():
    """A custom processor's hooks fire in order, it runs after the built-in
    bias stage, and before temperature scaling."""
    proc = RecordingProcessor(None, None)
    sampler = _make_sampler(proc)
    sampler.add_request(
        0,
        SamplingParams(logit_bias={TARGET_TOKEN: BIAS}, temperature=TEMPERATURE),
    )
    sampler.apply_staged_writes()

    num_reqs = 1
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    raw = torch.zeros(num_reqs, VOCAB_SIZE, device=DEVICE)
    out = sampler.apply_sampling_params(
        raw,
        expanded_idx_mapping=idx_mapping,
        idx_mapping=idx_mapping,
        idx_mapping_np=np.arange(num_reqs, dtype=np.intp),
        pos=torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE),
        input_ids=torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE),
        expanded_local_pos=torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE),
        seq_lens_upper_bound_np=np.ones(num_reqs, dtype=np.int64),
    )

    assert proc.events == [
        "add_request:0",
        "apply_staged_writes",
        "apply",
    ]
    # The processor sees the built-in bias applied but no temperature scaling.
    assert proc.seen_logits is not None
    assert proc.seen_logits[0, TARGET_TOKEN].item() == BIAS
    # Its own modification is then temperature-scaled by the sampler.
    assert out[0, TARGET_TOKEN].item() == (BIAS + BOOST) / TEMPERATURE


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("repetition", [1.0, 2.0])
@pytest.mark.parametrize("temperature", [0.0, 1.0])
@pytest.mark.parametrize(
    "mode", ["raw_logits", "raw_logprobs", "processed_logits", "processed_logprobs"]
)
def test_bias_centering_sampler_masks_penalties_and_scores(
    center, repetition, temperature, mode
):
    sampler = _make_sampler()
    sampler.logprobs_mode = mode
    states = sampler.req_states
    states.prompt_len.np[:2] = 1
    states.prefill_len.np[:2] = 1
    for row in range(2):
        states.all_token_ids.stage_write(row, 0, [0])
    states.apply_staged_writes()
    sampler.logits_processors[0] = LogitBiasState(
        None,
        LogitsProcRequestState.from_request_state(states),
        _center_logit_bias=center,
    )
    for row in range(2):
        sampler.add_request(
            row,
            SamplingParams(
                temperature=temperature,
                seed=42,
                logprobs=2,
                logit_bias={0: 0.25} if row == 0 else None,
                repetition_penalty=repetition if row == 0 else 1.0,
                allowed_token_ids=[0, 1],
            ),
        )
    sampler.apply_staged_writes()
    mapping = torch.arange(2, dtype=torch.int32, device=DEVICE)
    zeros = torch.zeros(2, dtype=torch.int32, device=DEVICE)
    # Minimal decode batch: two independent requests with one logit row each.
    batch = SimpleNamespace(
        expanded_idx_mapping=mapping,
        idx_mapping=mapping,
        idx_mapping_np=np.arange(2, dtype=np.intp),
        cu_num_logits_np=np.arange(3, dtype=np.int32),
        cu_num_logits=torch.arange(3, dtype=torch.int32, device=DEVICE),
        expanded_local_pos=zeros,
        positions=zeros,
        input_ids=zeros,
        logits_indices=mapping.to(torch.int64),
        seq_lens_cpu_upper_bound=torch.ones(2, dtype=torch.int32),
        seq_lens=torch.ones(2, dtype=torch.int32, device=DEVICE),
        num_reqs=2,
    )
    raw = torch.full((2, VOCAB_SIZE), 100.0, device=DEVICE)
    raw[:, :2] = torch.tensor([17.0, 16.0], device=DEVICE)
    saved = raw.clone()
    result = sampler(raw, batch)
    assert torch.equal(raw, saved)
    processed = torch.full_like(raw, -torch.inf)
    processed[1, :2] = raw[1, :2]
    processed[0, :2] = torch.tensor(
        [0.25 / repetition, -1.0] if center else [17.25 / repetition, 16.0],
        device=DEVICE,
    )
    expected = raw if mode.startswith("raw") else processed
    if mode.endswith("logprobs"):
        expected = expected.log_softmax(-1)
    scores = result.logprobs_tensors
    torch.testing.assert_close(
        scores.logprobs,
        expected.gather(-1, scores.logprob_token_ids.long()),
        rtol=1e-5,
        atol=1e-6,
    )
    assert (result.sampled_token_ids < 2).all()
    if temperature == 0:
        assert result.sampled_token_ids.flatten().tolist() == [
            int(repetition == 2.0 and not center),
            0,
        ]
    else:
        # Seeds and positions are fixed; replay must select the same legal tokens.
        assert torch.equal(
            result.sampled_token_ids, sampler(raw, batch).sampled_token_ids
        )
