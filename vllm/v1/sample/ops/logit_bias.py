# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch


class LogitBiasCentering:
    """Experimental pre-bias centering; callers must explicitly opt in.

    Eligibility is evaluated at the bias stage, after preceding masks. Rows
    whose finite range overflows FP32 retain their original scores.
    """

    def __init__(
        self,
        biases: dict[int, dict[int, float]],
        device: torch.device,
        is_pin_memory: bool = False,
    ):
        rows: list[int] = []
        bias_rows: list[int] = []
        bias_tokens: list[int] = []
        for row, bias in biases.items():
            # Match the precision of the bias tensor, including underflow to zero.
            tokens = [
                token
                for token, value in bias.items()
                if np.isfinite(np.float32(value)) and np.float32(value) != 0
            ]
            if tokens:
                bias_rows.extend([len(rows)] * len(tokens))
                bias_tokens.extend(tokens)
                rows.append(row)
        self.rows, self.bias_rows, self.bias_tokens = (
            torch.tensor(
                indices, dtype=torch.long, device="cpu", pin_memory=is_pin_memory
            ).to(device, non_blocking=True)
            for indices in (rows, bias_rows, bias_tokens)
        )

    def apply(self, logits: torch.Tensor) -> None:
        if self.rows.numel() == 0:
            return
        values = logits[self.rows]
        finite = torch.isfinite(values)
        eligible_biases = torch.zeros(
            self.rows.numel(), dtype=torch.int32, device=logits.device
        )
        eligible_biases.scatter_add_(
            0, self.bias_rows, finite[self.bias_rows, self.bias_tokens].to(torch.int32)
        )
        maximum = values.masked_fill(~finite, -torch.inf).amax(dim=-1, keepdim=True)
        minimum = values.masked_fill(~finite, torch.inf).amin(dim=-1, keepdim=True)
        center = (eligible_biases[:, None] > 0) & torch.isfinite(minimum - maximum)
        shift = torch.where(center, maximum, 0.0)
        logits[self.rows] = torch.where(center & finite, values - shift, values)
