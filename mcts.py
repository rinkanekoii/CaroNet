import math
import time
import collections
import itertools
from typing import Dict, List, Tuple, Set, NamedTuple, Optional

import numpy as np
import torch
from numba import njit

# Global Zobrist table for 15x15 board (0: player 1, 1: player -1)
ZOBRIST_TABLE = np.random.randint(-2**63, 2**63 - 1, size=(2, 256, 256), dtype=np.int64)

@njit(cache=True, nogil=True)
def _get_zobrist_hash_impl(board: np.ndarray, table: np.ndarray) -> int:
    h = np.int64(0)
    N = board.shape[0]
    for r in range(N):
        for c in range(N):
            v = board[r, c]
            if v == 1:
                h ^= table[0, r, c]
            elif v == -1:
                h ^= table[1, r, c]
    return h


def _get_zobrist_hash(board: np.ndarray) -> int:
    return int(_get_zobrist_hash_impl(board, ZOBRIST_TABLE))

class LeafInfo(NamedTuple):
    board: np.ndarray
    player: int
    node: "MCTSNode"
    path_nodes_buf: list
    path_indices: np.ndarray
    path_len: int
    zobrist_hash: int


from utils import check_win_adaptive, state_to_tensor_out

_LEGAL_MASK_CHANNEL = 2  # state_to_tensor_out layout: [player, opponent, empty, row?, col?]

# ─────────────────────────────────────────────────────────────────────
#  Numba-accelerated helpers
# ─────────────────────────────────────────────────────────────────────

@njit(cache=True, nogil=True)
def _has_any_neighbor(board, r: int, c: int) -> bool:
    """Check if cell (r,c) is adjacent to ANY piece (player or opponent)."""
    N = board.shape[0]
    r_lo = max(0, r - 1)
    r_hi = min(N, r + 2)
    c_lo = max(0, c - 1)
    c_hi = min(N, c + 2)
    for nr in range(r_lo, r_hi):
        for nc in range(c_lo, c_hi):
            if nr == r and nc == c:
                continue
            if board[nr, nc] != 0:
                return True
    return False

@njit(cache=True, nogil=True)
def _find_tactical_masks(board, player: int, win_length: int):
    N = board.shape[0]
    opponent = -player
    winning = np.zeros((N, N), dtype=np.uint8)
    blocking = np.zeros((N, N), dtype=np.uint8)

    for r in range(N):
        for c in range(N):
            if board[r, c] != 0:
                continue
            if not _has_any_neighbor(board, r, c):
                continue

            board[r, c] = player
            if check_win_adaptive(board, r, c, player, win_length):
                winning[r, c] = 1
            board[r, c] = 0

            board[r, c] = opponent
            if check_win_adaptive(board, r, c, opponent, win_length):
                blocking[r, c] = 1
            board[r, c] = 0

    return winning, blocking

@njit(cache=True, nogil=True)
def _find_fork_mask(board, player: int, win_length: int):
    """Find moves that create ≥2 winning threats.

    Optimised: only checks cells within `win_length` radius of the
    candidate rather than scanning the full board.
    """
    N = board.shape[0]
    mask = np.zeros((N, N), dtype=np.uint8)
    wl = win_length

    for r in range(N):
        for c in range(N):
            if board[r, c] != 0:
                continue
            if not _has_any_neighbor(board, r, c):
                continue

            board[r, c] = player
            threats = 0

            # Only scan within win_length radius — any winning line
            # through (r,c) must touch cells within this distance.
            tr_lo = max(0, r - wl + 1)
            tr_hi = min(N, r + wl)
            tc_lo = max(0, c - wl + 1)
            tc_hi = min(N, c + wl)

            for tr in range(tr_lo, tr_hi):
                for tc in range(tc_lo, tc_hi):
                    if tr == r and tc == c:
                        continue
                    if board[tr, tc] != 0:
                        continue
                    board[tr, tc] = player
                    if check_win_adaptive(board, tr, tc, player, win_length):
                        threats += 1
                    board[tr, tc] = 0
                    if threats >= 2:
                        break
                if threats >= 2:
                    break

            if threats >= 2:
                mask[r, c] = 1
            board[r, c] = 0
    return mask


@njit(cache=True, nogil=True)
def _select_child_puct(child_N, child_W, child_VL, child_P,
                       pw_threshold, c_puct, total_visits):
    """Vectorised PUCT selection in Numba — avoids numpy slice allocation."""
    best_idx = 0
    best_score = -1e30
    sqrt_sum = math.sqrt(max(1.0, float(total_visits)))

    for i in range(pw_threshold):
        n_i = child_N[i]
        vl_i = child_VL[i]
        denom = max(1, n_i + vl_i)
        q = (child_W[i] - float(vl_i)) / float(denom)
        u = c_puct * child_P[i] * (sqrt_sum / float(denom))
        score = q + u
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


@njit(cache=True, nogil=True)
def _compute_pw_threshold(total_visits, num_children, pw_alpha, pw_min):
    """Progressive widening threshold."""
    pw = max(pw_min, int(pw_alpha * math.sqrt(total_visits + 1)))
    if pw > num_children:
        pw = num_children
    return pw


# ─────────────────────────────────────────────────────────────────────

__all__ = [
    "MCTSNode",
    "ProgressiveMCTS",
    "find_tactical_moves",
    "find_forks",
    "distribute_forced_visits",
    "ONNXNetWrapper",
]

class ONNXNetWrapper:
    __slots__ = ("is_onnx", "session", "input_name")

    def __init__(self, onnx_path):
        import onnxruntime as ort
        ort.set_default_logger_severity(4)
        self.is_onnx = True
        available_providers = ort.get_available_providers()
        providers = []
        if "TensorrtExecutionProvider" in available_providers:
            providers.append((
                "TensorrtExecutionProvider",
                {
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": "./trt_cache",
                },
            ))
        if "CUDAExecutionProvider" in available_providers:
            providers.append((
                "CUDAExecutionProvider",
                {
                    "cudnn_conv_algo_search": "DEFAULT",
                    "arena_extend_strategy": "kSameAsRequested",
                },
            ))
        providers.append("CPUExecutionProvider")
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 4
        sess_options.log_verbosity_level = -1
        self.session = ort.InferenceSession(onnx_path, providers=providers, sess_options=sess_options)
        self.input_name = self.session.get_inputs()[0].name

    def eval(self):
        pass

    def __call__(self, numpy_states):
        import torch
        is_tensor = isinstance(numpy_states, torch.Tensor)
        device = numpy_states.device if is_tensor else None
        if is_tensor:
            numpy_states = numpy_states.detach().cpu().numpy()
        inputs = {self.input_name: numpy_states}
        v_preds, p_logits = self.session.run(None, inputs)
        if is_tensor:
            return torch.from_numpy(v_preds).to(device), torch.from_numpy(p_logits).to(device)
        return v_preds, p_logits

def find_tactical_moves(board: np.ndarray, player: int, win_length: int = 5) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    win_mask, block_mask = _find_tactical_masks(board, int(player), int(win_length))
    winning_moves = set()
    blocking_moves = set()
    wr = np.argwhere(win_mask != 0)
    for i in range(len(wr)):
        winning_moves.add((int(wr[i, 0]), int(wr[i, 1])))
    br = np.argwhere(block_mask != 0)
    for i in range(len(br)):
        blocking_moves.add((int(br[i, 0]), int(br[i, 1])))
    return winning_moves, blocking_moves

def find_forks(board: np.ndarray, player: int, win_length: int = 5) -> Set[Tuple[int, int]]:
    mask = _find_fork_mask(board, int(player), int(win_length))
    result = set()
    ar = np.argwhere(mask != 0)
    for i in range(len(ar)):
        result.add((int(ar[i, 0]), int(ar[i, 1])))
    return result

def distribute_forced_visits(moves, num_sims: int) -> Dict[Tuple[int, int], int]:
    moves = list(moves)
    if not moves:
        return {}
    n = len(moves)
    total = max(num_sims, n)
    base = total // n
    remainder = total - base * n
    counts = {}
    for i, move in enumerate(moves):
        counts[move] = base + (1 if i < remainder else 0)
    return counts

# ─────────────────────────────────────────────────────────────────────
#  MCTSNode – struct-of-arrays child storage
# ─────────────────────────────────────────────────────────────────────

class MCTSNode:
    __slots__ = (
        "P", "N",
        "child_moves", "child_P", "child_N", "child_W", "child_VL", "child_nodes",
        "is_expanded", "noise_applied", "num_children"
    )

    def __init__(self, prior: float = 0.0):
        self.P = prior
        self.N = 0
        self.is_expanded = False
        self.noise_applied = False
        self.num_children = 0
        self.child_moves = None
        self.child_P = None
        self.child_N = None
        self.child_W = None
        self.child_VL = None
        self.child_nodes = None

# ─────────────────────────────────────────────────────────────────────
#  ProgressiveMCTS
# ─────────────────────────────────────────────────────────────────────

class ProgressiveMCTS:
    __slots__ = (
        "net", "board_size", "device", "c_puct", "noise_eps",
        "batch_size", "win_length", "progressive_widening", "pw_alpha", "pw_min",
        "dir_alpha", "root", "use_coords",
        "_state_channels", "_states_buffer", "_scratch_board",
        "_is_onnx", "_autocast_fn", "_device_type",
        "_board_cells", "_inference_cache"
    )

    def __init__(self, net, board_size: int = 20, device: str = "cpu",
                 c_puct: float = 2.5, dirichlet_alpha=None,
                 noise_eps: float = 0.35, batch_size: int = 64,
                 win_length: int = 5, progressive_widening: bool = True,
                 use_coords: bool = False, pw_alpha: float = 3.0, pw_min: int = 5):
        self.net = net
        if board_size > ZOBRIST_TABLE.shape[1]:
            raise ValueError(
                f"ZOBRIST_TABLE supports board_size <= {ZOBRIST_TABLE.shape[1]}, got {board_size}"
            )
        self.board_size = board_size
        self.device = device
        self.c_puct = c_puct
        self.noise_eps = noise_eps
        self.batch_size = batch_size
        self.win_length = win_length
        self.progressive_widening = progressive_widening
        self.pw_alpha = pw_alpha
        self.pw_min = pw_min
        self.dir_alpha = 15.0 / board_size if dirichlet_alpha is None else dirichlet_alpha
        self.root = MCTSNode()
        self.use_coords = use_coords
        self._state_channels = 3 + (2 if use_coords else 0)
        self._states_buffer = np.empty(
            (self.batch_size, self._state_channels, self.board_size, self.board_size),
            dtype=np.float32,
        )
        self._scratch_board = np.empty((self.board_size, self.board_size), dtype=np.int8)
        self._is_onnx = getattr(net, "is_onnx", False)
        self._board_cells = board_size * board_size

        self._inference_cache = collections.OrderedDict()

        # Pre-resolve autocast once
        self._device_type = ""
        self._autocast_fn = None
        if not self._is_onnx:
            dt = device.type if hasattr(device, "type") else str(device).split(":")[0]
            self._device_type = dt
            if dt == "cuda":
                try:
                    self._autocast_fn = torch.amp.autocast
                except AttributeError:
                    self._autocast_fn = torch.cuda.amp.autocast

    def reset_root(self):
        self.root = MCTSNode()

    def move_root(self, mv):
        root = self.root
        if root and root.child_moves is not None:
            cm = root.child_moves
            # Fast search: child_moves is (K,2) int32
            matches = np.where((cm[:, 0] == mv[0]) & (cm[:, 1] == mv[1]))[0]
            if len(matches) > 0:
                idx = matches[0]
                if root.child_nodes[idx] is None:
                    root.child_nodes[idx] = MCTSNode(prior=root.child_P[idx])
                new_root = root.child_nodes[idx]
                if new_root.child_VL is not None:
                    new_root.child_VL.fill(0)
                self.root = new_root
                # Keep the child's real expansion state. Forcing True here can
                # waste the first simulation after a move, and with tiny sims it
                # can make the next policy effectively uniform. Software: where
                # one boolean can quietly waste a search.
                self.root.noise_applied = False
                return True
        self.reset_root()
        return False

    def _set_node_children(self, node: MCTSNode, move_indices: np.ndarray,
                           probs: np.ndarray, board_size: int):
        """Set children from flat index + probability arrays (already filtered/sorted)."""
        num_children = len(move_indices)
        if num_children == 0:
            return

        # Sort by probability descending
        sort_idx = np.argsort(-probs)
        move_indices = move_indices[sort_idx]
        probs = probs[sort_idx]

        rows, cols = np.divmod(move_indices, board_size)
        node.child_moves = np.stack([rows, cols], axis=1).astype(np.int32)
        node.child_P = probs.astype(np.float32)
        node.child_N = np.zeros(num_children, dtype=np.int32)
        node.child_W = np.zeros(num_children, dtype=np.float32)
        node.child_VL = np.zeros(num_children, dtype=np.int32)
        node.child_nodes = [None] * num_children
        node.num_children = num_children
        node.is_expanded = True

    def expand_leaf_nodes_batch(self, leaf_list: List[LeafInfo]):
        if not leaf_list:
            return []
        
        board_size = self.board_size
        batch_size = self.batch_size
        use_coords = self.use_coords

        cached_results = []
        boards_to_infer = []

        # 1. Soft TT Interception
        for leaf in leaf_list:
            key = (leaf.zobrist_hash, leaf.player)
            if key in self._inference_cache:
                self._inference_cache.move_to_end(key)
                cached_results.append((leaf, self._inference_cache[key]))
            else:
                boards_to_infer.append((leaf, key))

        # 2. Neural network inference for non-cached boards
        bs = len(boards_to_infer)
        if bs > 0:
            if bs <= batch_size:
                states = self._states_buffer[:bs]
            else:
                states = np.empty(
                    (bs, self._state_channels, board_size, board_size),
                    dtype=np.float32,
                )

            for i in range(bs):
                leaf = boards_to_infer[i][0]
                state_to_tensor_out(leaf.board, leaf.player, states[i], use_coords=use_coords)

            # ── Neural network inference ──
            if self._is_onnx:
                v_preds, p_logits = self.net(states)
                v_vals = np.asarray(v_preds, dtype=np.float32).ravel()
                p_logits = np.asarray(p_logits, dtype=np.float32)
            else:
                self.net.eval()
                x = torch.from_numpy(states)
                device_type = self._device_type
                if device_type == "cuda":
                    x = x.pin_memory().to(self.device, non_blocking=True).contiguous(
                        memory_format=torch.channels_last
                    )
                else:
                    x = x.to(self.device)

                with torch.inference_mode():
                    if device_type == "cuda" and self._autocast_fn is not None:
                        try:
                            ctx = self._autocast_fn("cuda", enabled=True)
                        except TypeError:
                            ctx = self._autocast_fn(enabled=True)
                        with ctx:
                            v_preds, p_logits = self.net(x)
                    else:
                        v_preds, p_logits = self.net(x)

                    v_vals = v_preds.float().cpu().numpy().ravel()
                    p_logits = p_logits.float().cpu().numpy()

            # ── Compute policy probabilities (vectorized over batch) ──
            masks = states[:bs, _LEGAL_MASK_CHANNEL].reshape(bs, -1)
            legal_logits = np.where(masks > 0, p_logits[:bs].reshape(bs, -1), -np.inf)

            # Numerically stable softmax
            max_logits = np.max(legal_logits, axis=1, keepdims=True)
            max_logits = np.where(np.isinf(max_logits), 0.0, max_logits)
            exp_logits = np.exp(legal_logits - max_logits)
            exp_logits *= masks  # zero out illegal
            sums = exp_logits.sum(axis=1, keepdims=True)

            # Handle zero-sum rows
            zero_mask = (sums.ravel() <= 1e-8)
            if np.any(zero_mask):
                exp_logits[zero_mask] = masks[zero_mask]
                sums[zero_mask] = exp_logits[zero_mask].sum(axis=1, keepdims=True)

            # In-place normalize
            sums_inv = np.float32(1.0) / (sums + np.float32(1e-8))
            probs_all = exp_logits * sums_inv

            # Save to Cache
            for i in range(bs):
                leaf, key = boards_to_infer[i]
                v = float(v_vals[i])
                p = probs_all[i].copy()
                m = masks[i].copy()
                self._inference_cache[key] = (v, p, m)
                cached_results.append((leaf, (v, p, m)))
            
            # Simple FIFO-like clearing to prevent OOM
            if len(self._inference_cache) > 200000:
                # Remove oldest 50,000 items
                keys_to_delete = list(itertools.islice(self._inference_cache, 50000))
                for k in keys_to_delete:
                    del self._inference_cache[k]

        # 3. Apply results to all leaves
        results = []
        for leaf, (v, p, m) in cached_results:
            node = leaf.node
            legal_indices = np.flatnonzero(m)

            if len(legal_indices) == 0:
                results.append((leaf, 0.0))
                continue

            legal_probs = p[legal_indices]
            self._set_node_children(node, legal_indices, legal_probs, board_size)
            results.append((leaf, v))

        return results

    def backup_path(self, root_node, path_nodes_buf, path_indices, path_len, leaf_value):
        """Back-propagate value through the path."""
        root_node.N += 1

        v = leaf_value
        for k in range(path_len - 1, -1, -1):
            node = path_nodes_buf[k]
            idx = path_indices[k]
            node.child_N[idx] += 1
            node.child_W[idx] += v
            node.child_VL[idx] -= 1
            child = node.child_nodes[idx]
            if child is not None:
                child.N += 1
            v = -v

    def run_simulations(
        self, board: np.ndarray, player: int, num_sims: int = 200, add_noise: bool = True, time_limit_ms: Optional[float] = None
    ):
        start_time = time.perf_counter()
        root = self.root
        win_length = self.win_length
        opponent = -player
        board_size = self.board_size
        total_cells = self._board_cells

        # ── Tactical short-circuits ──
        winning_moves, blocking_moves = find_tactical_moves(board, player, win_length)
        if winning_moves:
            return {next(iter(winning_moves)): num_sims}

        if blocking_moves:
            if len(blocking_moves) == 1:
                return {next(iter(blocking_moves)): num_sims}
            return distribute_forced_visits(blocking_moves, num_sims)

        my_forks = find_forks(board, player, win_length)
        if my_forks:
            if len(my_forks) == 1:
                return {next(iter(my_forks)): num_sims}
            return distribute_forced_visits(my_forks, num_sims)

        opp_forks = find_forks(board, opponent, win_length)
        if opp_forks:
            if len(opp_forks) == 1:
                return {next(iter(opp_forks)): num_sims}
            return distribute_forced_visits(opp_forks, num_sims)

        # ── Expand root if needed ──
        if not root.is_expanded:
            z_hash = _get_zobrist_hash(board)
            leaf = LeafInfo(board, player, root, [], np.empty(0, dtype=np.int32), 0, z_hash)
            self.expand_leaf_nodes_batch([leaf])

        # ── Root noise injection ──
        if (add_noise and root.child_moves is not None
                and root.num_children > 0 and not root.noise_applied):
            nc = root.num_children
            noise = np.random.dirichlet([self.dir_alpha] * nc).astype(np.float32)
            eps = np.float32(self.noise_eps)
            root.child_P = (np.float32(1.0) - eps) * root.child_P + eps * noise
            sort_idx = np.argsort(-root.child_P)
            root.child_moves = root.child_moves[sort_idx]
            root.child_P = root.child_P[sort_idx]
            root.child_N = root.child_N[sort_idx]
            root.child_W = root.child_W[sort_idx]
            root.child_VL = root.child_VL[sort_idx]
            root.child_nodes = [root.child_nodes[i] for i in sort_idx]
            root.noise_applied = True

        # ── Simulation loop ──
        # Cache hot attributes as locals
        c_puct = float(self.c_puct)
        pw_enabled = self.progressive_widening
        scratch_board = self._scratch_board
        batch_size = self.batch_size
        backup_path = self.backup_path

        # Pre-allocate path storage (max depth = total_cells)
        max_depth = total_cells
        path_indices_pool = np.empty((batch_size, max_depth), dtype=np.int32)
        path_nodes_pool = [[None] * max_depth for _ in range(batch_size)]

        leaf_batch = []
        sims_done = 0
        base_filled = int(np.count_nonzero(board))

        while sims_done + len(leaf_batch) < num_sims:
            node = root
            np.copyto(scratch_board, board)
            cur_board = scratch_board
            filled_cells = base_filled
            cur_player = player
            terminal = False

            path_len = 0

            while node.is_expanded and node.num_children > 0:
                total_visits = node.N
                num_children = node.num_children

                if pw_enabled:
                    pw_threshold = _compute_pw_threshold(total_visits, num_children, self.pw_alpha, self.pw_min)
                else:
                    pw_threshold = num_children

                best_idx = _select_child_puct(
                    node.child_N, node.child_W, node.child_VL, node.child_P,
                    pw_threshold, c_puct, total_visits,
                )

                node.child_VL[best_idx] += 1
                child = node.child_nodes[best_idx]
                if child is None:
                    child = MCTSNode(prior=node.child_P[best_idx])
                    node.child_nodes[best_idx] = child

                path_nodes_pool[len(leaf_batch)][path_len] = node
                path_indices_pool[len(leaf_batch)][path_len] = best_idx
                path_len += 1

                r = node.child_moves[best_idx, 0]
                c = node.child_moves[best_idx, 1]
                if cur_board[r, c] != 0:
                    raise RuntimeError(
                        f"Mcripts selected illegal child move ({int(r)}, {int(c)}); "
                        f"cell already contains {int(cur_board[r, c])}. Root/tree is stale or legal mask is corrupt."
                    )
                cur_board[r, c] = cur_player
                filled_cells += 1

                if check_win_adaptive(cur_board, r, c, cur_player, win_length):
                    backup_path(root, path_nodes_pool[len(leaf_batch)], path_indices_pool[len(leaf_batch)], path_len, 1.0)
                    terminal = True
                    break
                if filled_cells >= total_cells:
                    backup_path(root, path_nodes_pool[len(leaf_batch)], path_indices_pool[len(leaf_batch)], path_len, 0.0)
                    terminal = True
                    break

                cur_player = -cur_player
                node = child

            if terminal:
                sims_done += 1
                continue

            leaf_batch.append(LeafInfo(
                board=cur_board.copy(),
                player=cur_player,
                node=node,
                path_nodes_buf=path_nodes_pool[len(leaf_batch)],
                path_indices=path_indices_pool[len(leaf_batch)],
                path_len=path_len,
                zobrist_hash=_get_zobrist_hash(cur_board),
            ))

            if len(leaf_batch) >= batch_size:
                batch_results = self.expand_leaf_nodes_batch(leaf_batch)
                for leaf, v_pred in batch_results:
                    leaf_value = -float(v_pred)
                    backup_path(
                        root,
                        leaf.path_nodes_buf,
                        leaf.path_indices,
                        leaf.path_len,
                        leaf_value,
                    )
                sims_done += len(leaf_batch)
                leaf_batch = []

        # Flush any remaining leaves (happens if the loop exits via terminal simulation `continue`)
        if len(leaf_batch) > 0:
            batch_results = self.expand_leaf_nodes_batch(leaf_batch)
            for leaf, v_pred in batch_results:
                leaf_value = -float(v_pred)
                backup_path(
                    root,
                    leaf.path_nodes_buf,
                    leaf.path_indices,
                    leaf.path_len,
                    leaf_value,
                )
            sims_done += len(leaf_batch)

        if root.child_moves is not None:
            return {(int(mv[0]), int(mv[1])): int(n) for mv, n in zip(root.child_moves, root.child_N)}
        return {}

    def get_policy(self, tau: float = 1.0) -> Dict[Tuple[int, int], float]:
        root = self.root
        if root.child_moves is None or root.num_children == 0:
            return {}
        
        visits = root.child_N.astype(np.float32)
        total_visits = float(visits.sum())
        if total_visits <= 0:
            probs = np.ones(root.num_children, dtype=np.float32) / root.num_children
        elif tau < 0.01:
            best_idx = np.argmax(visits)
            probs = np.zeros(root.num_children, dtype=np.float32)
            probs[best_idx] = 1.0
        else:
            inv_tau = 1.0 / tau
            probs = visits ** inv_tau
            p_sum = float(probs.sum())
            if p_sum > 0:
                probs /= p_sum
            else:
                probs = np.ones(root.num_children, dtype=np.float32) / root.num_children
                
        return {(int(mv[0]), int(mv[1])): float(p) for mv, p in zip(root.child_moves, probs)}
