# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare default and experimental logit-bias sampling on a configured CUDA build.

Run correctness tests first, then:
    .venv/bin/python benchmarks/kernels/benchmark_logit_bias.py --runner v1
    .venv/bin/python benchmarks/kernels/benchmark_logit_bias.py --runner v2

Times the sampling pipeline, including its allocations and a common input copy.
Request setup, compilation and input generation are excluded. GPU measurements
exclude Python dispatch; wall measurements include it. CUPTI is the default;
use --timer cuda-graph when CUPTI 13+ is unavailable. Both use cold L2 inputs.
No logprobs, penalties or top-k/top-p are requested. Outputs are JSON lines.
Profiler events are best-effort; compare launch API calls as well as GPU events.
"""

import argparse
import itertools
import json
import statistics
import subprocess
import time
import warnings
from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch
from flashinfer.testing import bench_gpu_time_with_cudagraph, bench_gpu_time_with_cupti

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitBiasLogitsProcessor,
    LogitsProcessors,
)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler as V1Sampler
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.logits_processor import LogitsProcRequestState
from vllm.v1.worker.gpu.sample.sampler import Sampler as V2Sampler
from vllm.v1.worker.gpu.states import RequestState


def make_runner(runner, raw, params, center):
    batch, vocab = raw.shape
    device = raw.device
    experimental = {"_center_logit_bias": True} if center else {}
    if runner == "v1":
        processor = LogitBiasLogitsProcessor(None, device, False, **experimental)
        processor.update_state(
            BatchUpdate(batch, [], [(i, p, None, []) for i, p in enumerate(params)], [])
        )
        greedy = params[0].temperature == 0
        metadata = SamplingMetadata(
            temperature=torch.full((batch,), params[0].temperature, device=device),
            all_greedy=greedy,
            all_random=not greedy,
            top_p=None,
            top_k=None,
            generators={},
            max_num_logprobs=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=torch.zeros(batch, device=device),
            presence_penalties=torch.zeros(batch, device=device),
            repetition_penalties=torch.ones(batch, device=device),
            output_token_ids=[[] for _ in params],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors([processor]),
        )
        sampler = V1Sampler()
        return lambda logits: sampler(logits.clone(), metadata).sampled_token_ids

    states = RequestState(batch, 16, batch, 1, vocab, device)
    states.prompt_len.np[:] = 1
    states.prefill_len.np[:] = 1
    states.apply_staged_writes()
    sampler = V2Sampler(
        SimpleNamespace(reasoning_config=None), batch, vocab, device, states
    )
    sampler.logits_processors[0] = LogitBiasState(
        None,
        LogitsProcRequestState.from_request_state(states),
        **experimental,
    )
    for i, param in enumerate(params):
        sampler.add_request(i, param)
    sampler.apply_staged_writes()
    mapping = torch.arange(batch, dtype=torch.int32, device=device)
    mapping_np = np.arange(batch, dtype=np.intp)
    zeros = torch.zeros(batch, dtype=torch.int32, device=device)
    lengths = np.ones(batch, dtype=np.int64)
    return lambda logits: sampler.sample(
        logits.clone(), mapping, mapping, mapping_np, zeros, zeros, zeros, lengths
    )[0]


def measure(run, raw, args):
    for _ in range(20):
        run(raw)
    torch.accelerator.synchronize()
    torch.accelerator.reset_peak_memory_stats()
    allocated = torch.accelerator.memory_allocated()
    run(raw)
    torch.accelerator.synchronize()
    peak_bytes = torch.accelerator.max_memory_allocated() - allocated
    timer = (
        bench_gpu_time_with_cupti
        if args.timer == "cupti"
        else bench_gpu_time_with_cudagraph
    )
    options = {"use_cuda_graph": True} if args.timer == "cupti" else {}
    with warnings.catch_warnings():
        # Do not label an automatic event-timing fallback as a CUPTI result.
        warnings.filterwarnings("error", message=".*Falling back to CUDA events.*")
        trials_us = [
            1000
            * statistics.median(
                timer(
                    run,
                    input_args=(raw,),
                    cold_l2_cache=True,
                    dry_run_iters=5,
                    repeat_iters=args.repeat,
                    **options,
                )
            )
            for _ in range(args.trials)
        ]
    wall_us = []
    for _ in range(args.trials):
        torch.accelerator.synchronize()
        start = time.perf_counter()
        for _ in range(args.repeat):
            run(raw)
        torch.accelerator.synchronize()
        wall_us.append((time.perf_counter() - start) * 1e6 / args.repeat)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        run(raw)
        torch.accelerator.synchronize()
    cuda_events = Counter(
        event.name
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    )
    launch_calls = Counter(
        event.name
        for event in prof.events()
        if event.name.startswith(("cudaLaunch", "cuLaunch", "cudaGraphLaunch"))
    )
    return dict(
        gpu_us=statistics.median(trials_us),
        gpu_trials_us=trials_us,
        wall_us=statistics.median(wall_us),
        wall_trials_us=wall_us,
        peak_temporary_bytes=peak_bytes,
        cuda_events=sum(cuda_events.values()) if cuda_events else None,
        cuda_event_counts=dict(cuda_events),
        cuda_launch_counts=dict(launch_calls),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", choices=["v1", "v2"], required=True)
    parser.add_argument("--center", choices=["off", "on", "both"], default="both")
    parser.add_argument("--timer", choices=["cupti", "cuda-graph"], default="cupti")
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument("--vocab", type=int, default=128256)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    print(
        json.dumps(
            dict(
                gpu=torch.cuda.get_device_name(),
                torch=torch.__version__,
                cuda=torch.version.cuda,
                commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                settings=vars(args),
            )
        )
    )
    for batch, population, temperature in itertools.product(
        args.batch, ["none", "all", "mixed"], [0.0, 1.0]
    ):
        # A one-request batch cannot contain both biased and unbiased requests.
        if batch == 1 and population == "mixed":
            continue
        torch.manual_seed(42)
        raw = torch.randn(batch, args.vocab, device="cuda", dtype=torch.float32) + 17
        params = [
            SamplingParams(
                temperature=temperature,
                seed=42,
                logit_bias={0: 0.25, 1: -0.5}
                if population == "all" or (population == "mixed" and i % 2 == 0)
                else None,
            )
            for i in range(batch)
        ]
        variants = {"off": [False], "on": [True], "both": [False, True]}[args.center]
        for center in variants:
            run = make_runner(args.runner, raw, params, center)
            tokens = run(raw)
            assert ((tokens >= 0) & (tokens < args.vocab)).all()
            result = measure(run, raw, args)
            result.update(
                batch=batch,
                population=population,
                temperature=temperature,
                center=center,
                gpu_tokens_per_second=batch * 1e6 / result["gpu_us"],
                wall_tokens_per_second=batch * 1e6 / result["wall_us"],
            )
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
