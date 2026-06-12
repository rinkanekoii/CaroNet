import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from augmentation import augment_sample, random_augment_single

__all__ = ["PrioritizedReplayBuffer", "ReplayDataset", "augment_sample", "random_augment_single"]


class FenwickTree:
    """Prefix-sum tree for O(log n) weighted priority sampling."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.tree = np.zeros(self.capacity + 1, dtype=np.float64)
        self.values = np.zeros(self.capacity, dtype=np.float64)
        self._bit = 1 << (self.capacity.bit_length() - 1)

    def reset(self):
        self.tree.fill(0.0)
        self.values.fill(0.0)

    def update(self, idx: int, value: float):
        i = int(idx) + 1
        delta = float(value) - self.values[idx]
        self.values[idx] = float(value)
        while i <= self.capacity:
            self.tree[i] += delta
            i += i & -i

    def batch_update(self, indices: np.ndarray, values: np.ndarray):
        for idx, val in zip(indices, values):
            self.update(int(idx), float(val))

    def prefix_sum(self, idx: int) -> float:
        i = int(idx) + 1
        s = 0.0
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s


    def total(self, size: int | None = None) -> float:
        if size is None or size >= self.capacity:
            return float(self.prefix_sum(self.capacity - 1))
        if size <= 0:
            return 0.0
        return float(self.prefix_sum(size - 1))

    def find_prefix(self, mass: float) -> int:
        """Return smallest index whose prefix sum is greater than mass."""
        idx = 0
        bit = self._bit
        while bit:
            nxt = idx + bit
            if nxt <= self.capacity and self.tree[nxt] <= mass:
                idx = nxt
                mass -= self.tree[nxt]
            bit >>= 1
        return min(idx, self.capacity - 1)



import threading

class PrioritizedReplayBuffer:
    """Optimized replay buffer using pre-allocated numpy arrays for faster sampling."""

    def __init__(self, capacity: int = 200000, state_shape: tuple = None):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0
        self.state_shape = state_shape
        self._initialized = False
        self.states = None
        self.pis = None
        self.zs = None
        self.priorities = None
        self.priority_tree = None
        self._priority_max: float = 1.0
        self._priority_min: float = 1.0
        self._sample_count: int = 0
        self._lock = threading.Lock()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_lock'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()


    def _recompute_priority_bounds(self):
        if self.size == 0:
            return
        active = self.priorities[:self.size]
        self._priority_min = float(active.min())
        self._priority_max = float(active.max())

    def save(self, path: str):
        with self._lock:
            if not self._initialized:
                return
            np.savez_compressed(
                path,
                states=self.states[:self.size],
                pis=self.pis[:self.size],
                zs=self.zs[:self.size],
                priorities=self.priorities[:self.size],
                meta=np.array([self.ptr, self.size]),
            )

    @classmethod
    def load(cls, path: str, capacity: int) -> "PrioritizedReplayBuffer":
        data = np.load(path)
        buf = cls(capacity=capacity)
        buf.push_batch(data["states"], data["pis"], data["zs"], data["priorities"])
        buf.ptr = int(data["meta"][0])
        return buf

    def _init_buffers(self, state, pi):

        """Lazy initialization of buffers based on first sample."""
        self.state_shape = state.shape
        pi_size = pi.shape[0]
        self.states = np.zeros((self.capacity, *self.state_shape), dtype=np.float32)
        self.pis = np.zeros((self.capacity, pi_size), dtype=np.float32)
        self.zs = np.zeros(self.capacity, dtype=np.float32)
        self.priorities = np.ones(self.capacity, dtype=np.float32)
        self.priority_tree = FenwickTree(self.capacity)
        self._initialized = True

    def push(self, state, pi, z, priority: float = 1.0):
        with self._lock:
            self._push(state, pi, z, priority)

    def _push(self, state, pi, z, priority: float = 1.0):
        if not self._initialized:
            self._init_buffers(state, pi)

        self.states[self.ptr] = state
        self.pis[self.ptr] = pi
        self.zs[self.ptr] = z
        self.priorities[self.ptr] = priority
        self.priority_tree.update(self.ptr, float(priority))

        if priority > self._priority_max:
            self._priority_max = priority
        if priority < self._priority_min:
            self._priority_min = priority

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def push_batch(self, states, pis, zs, priorities=None):
        with self._lock:
            self._push_batch(states, pis, zs, priorities)

    def _push_batch(self, states, pis, zs, priorities=None):
        """Push multiple samples at once for efficiency."""
        batch_size = len(states)
        if batch_size == 0:
            return
        if not self._initialized:
            self._init_buffers(states[0], pis[0])

        if priorities is None:
            priorities = np.ones(batch_size, dtype=np.float32)

        states_arr = np.asarray(states, dtype=np.float32)
        pis_arr = np.asarray(pis, dtype=np.float32)
        zs_arr = np.asarray(zs, dtype=np.float32)
        prios_arr = np.asarray(priorities, dtype=np.float32)

        batch_max = float(prios_arr.max())
        batch_min = float(prios_arr.min())
        if batch_max > self._priority_max:
            self._priority_max = batch_max
        if batch_min < self._priority_min:
            self._priority_min = batch_min

        if batch_size >= self.capacity:
            states_arr = states_arr[-self.capacity:]
            pis_arr = pis_arr[-self.capacity:]
            zs_arr = zs_arr[-self.capacity:]
            prios_arr = prios_arr[-self.capacity:]
            batch_size = self.capacity
            self.ptr = 0
            self.size = self.capacity
            self.states[:] = states_arr
            self.pis[:] = pis_arr
            self.zs[:] = zs_arr
            self.priorities[:] = prios_arr
            self.priority_tree.reset()
            for idx, prio in enumerate(prios_arr):
                self.priority_tree.update(idx, float(prio))
            return

        end = self.ptr + batch_size
        if end <= self.capacity:
            self.states[self.ptr:end] = states_arr
            self.pis[self.ptr:end] = pis_arr
            self.zs[self.ptr:end] = zs_arr
            self.priorities[self.ptr:end] = prios_arr
        else:
            first = self.capacity - self.ptr
            self.states[self.ptr:] = states_arr[:first]
            self.pis[self.ptr:] = pis_arr[:first]
            self.zs[self.ptr:] = zs_arr[:first]
            self.priorities[self.ptr:] = prios_arr[:first]

            rest = batch_size - first
            self.states[:rest] = states_arr[first:]
            self.pis[:rest] = pis_arr[first:]
            self.zs[:rest] = zs_arr[first:]
            self.priorities[:rest] = prios_arr[first:]

        start = self.ptr
        indices = (start + np.arange(batch_size)) % self.capacity
        self.priority_tree.batch_update(indices, prios_arr)

        self.ptr = end % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        with self._lock:
            self._update_priorities(indices, priorities)

    def _update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """

        Args:
            indices:    Mảng indices trả về từ sample_indices().
            priorities: Mảng priority mới, thường = |TD-error| + epsilon.
        """
        if not self._initialized:
            return
        priorities = np.asarray(priorities, dtype=np.float32)
        np.clip(priorities, a_min=1e-6, a_max=None, out=priorities)
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio
            self.priority_tree.update(int(idx), float(prio))

        new_max = float(priorities.max())
        if new_max > self._priority_max:
            self._priority_max = new_max

        new_min = float(priorities.min())
        if new_min < self._priority_min:
            self._priority_min = new_min

    def _weighted_sample_indices_tree(self, n: int, replace: bool = False):
        """Draw priority-weighted indices in O(n log capacity)."""
        n = min(int(n), self.size)
        if n <= 0:
            return np.array([], dtype=np.int64)
        total = self.priority_tree.total(self.size) if self.priority_tree is not None else 0.0
        if total <= 0.0:
            return np.random.choice(self.size, size=n, replace=False)

        out = np.empty(n, dtype=np.int64)
        changed = []
        running_total = total
        for i in range(n):
            if running_total <= 0.0:
                out = out[:i]
                break
            for _retry in range(10):
                idx = self.priority_tree.find_prefix(np.random.random() * running_total)
                if idx < self.size:
                    break
            else:
                idx = np.random.randint(0, self.size)
            out[i] = idx
            if not replace:
                old = float(self.priorities[idx])
                changed.append((idx, old))
                self.priority_tree.update(idx, 0.0)
                running_total -= old  # O(1) thay vì O(log n)

        if changed:
            for idx, old in changed:
                self.priority_tree.update(idx, old)
        return out

    def sample(self, n: int):
        with self._lock:
            if self.size == 0:
                return None, None, None
            n = min(n, self.size)
            indices = self._sample_indices(n)
            return self.states[indices], self.pis[indices], self.zs[indices]

    def sample_with_indices(self, n: int, beta: float = 0.4):
        with self._lock:
            if self.size == 0:
                return None, None, None, None, None
            n = min(n, self.size)
            indices = self._sample_indices(n)
            
            # IS weights
            probs = self.priorities[indices] / self.priority_tree.total(self.size)
            is_weights = (self.size * probs) ** (-beta)
            is_weights = (is_weights / is_weights.max()).astype(np.float32)

            return self.states[indices], self.pis[indices], self.zs[indices], indices, is_weights

    def _sample_indices(self, n: int) -> np.ndarray:
        """Return sample indices. Weighted mode uses a Fenwick tree: O(log n) per draw."""
        if self.size == 0:
            return np.array([], dtype=np.int64)
        n = min(n, self.size)
        if self._priority_max - self._priority_min <= 1e-6:
            return np.random.choice(self.size, size=n, replace=False)
        return self._weighted_sample_indices_tree(n, replace=False)

    def __len__(self):
        return self.size

    def get_dataloader(
        self,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = True,
        augment: bool = True,
    ):
        """Create an efficient DataLoader for training with optional augmentation."""
        if self.size == 0:
            return None
        dataset = ReplayDataset(self, augment=augment)

        loader_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=num_workers > 0,
        )
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = 2

        use_weighted = (self._priority_max - self._priority_min) > 1e-6
        if use_weighted:
            from torch.utils.data import WeightedRandomSampler

            active_priorities = self.priorities[: self.size]
            weights = torch.as_tensor(active_priorities, dtype=torch.double)
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=self.size,
                replacement=True,              )
            return DataLoader(dataset, sampler=sampler, **loader_kwargs)

        return DataLoader(dataset, shuffle=True, **loader_kwargs)


class ReplayDataset(Dataset):
    """PyTorch Dataset wrapper for replay buffer with optional augmentation."""

    def __init__(self, buffer: PrioritizedReplayBuffer, augment: bool = True):
        self.buffer = buffer
        self.augment = augment
        if buffer.state_shape is not None:
            # state shape thường là (channels, H, W) — lấy H làm board_size
            self.board_size = buffer.state_shape[-1]
        elif buffer.pis is not None:
            self.board_size = int(np.sqrt(buffer.pis.shape[1]))
        else:
            self.board_size = 15

    def __len__(self):
        return self.buffer.size

    def __getitem__(self, idx):
        state = self.buffer.states[idx]
        pi = self.buffer.pis[idx]
        z = self.buffer.zs[idx]

        if self.augment:
            state, pi = random_augment_single(state, pi, self.board_size)

        s = np.ascontiguousarray(state)
        p = np.ascontiguousarray(pi)
        return (
            torch.from_numpy(s),
            torch.from_numpy(p),
            torch.tensor(z, dtype=torch.float32),
        )
