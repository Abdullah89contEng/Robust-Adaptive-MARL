"""D_exp / D_exe replay buffers (method-spec.md §9).

- D_exp: (o_i,t, a_i,t^exp, o_i,t+1) tagged with the active regime label
  mu_i(t). Drives the exploration policy + contrastive encoder.
- D_exe: (o_i,t, a_i,t^exe, o_i,t+1, r_i,t, mu_i(t)). Also receives every
  exploration transition with the intrinsic reward stripped, so
  execution-side context inference sees the full trajectory.

Never stored (always recomputed at sample time): intrinsic reward r_aux,
sampled contexts z_i, MAAL's check_a^-i. This module only provides the
generic flat storage; it's the training loop's job (not this buffer's) to
never pass those three fields in.
"""

from __future__ import annotations

import torch


class ReplayBuffer:
    """A flat, field-agnostic ring buffer.

    Each `add(**fields)` call ingests one batch of transitions — e.g. a
    single environment step across every (vectorized env, agent) pair,
    pre-flattened by the caller into a single leading "sample" dimension.
    Field shapes are inferred and fixed from the first `add()` call.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._storage: dict[str, torch.Tensor] = {}
        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, **fields: torch.Tensor) -> None:
        n = next(iter(fields.values())).shape[0]
        if not self._storage:
            for name, value in fields.items():
                self._storage[name] = torch.zeros((self.capacity, *value.shape[1:]), dtype=value.dtype)

        if n > self.capacity:
            fields = {k: v[-self.capacity :] for k, v in fields.items()}
            n = self.capacity

        end = self._ptr + n
        if end <= self.capacity:
            for name, value in fields.items():
                self._storage[name][self._ptr : end] = value.detach()
        else:
            first = self.capacity - self._ptr
            for name, value in fields.items():
                self._storage[name][self._ptr :] = value[:first].detach()
                self._storage[name][: end - self.capacity] = value[first:].detach()

        self._ptr = end % self.capacity
        self._size = min(self._size + n, self.capacity)

    def sample(self, batch_size: int, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        if self._size == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer")
        indices = torch.randint(0, self._size, (batch_size,), generator=generator)
        return {name: tensor[indices] for name, tensor in self._storage.items()}

    def is_ready(self, batch_size: int) -> bool:
        return self._size >= batch_size
