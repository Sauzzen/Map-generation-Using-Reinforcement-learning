# ===============================
# map_env.py (improved version)
# ===============================

import numpy as np
from collections import deque

# Tile codes
LAND, WATER, ROCK, ROAD = 0, 1, 2, 3
NUM_TILES = 4
N = 10  # default chunk size (10x10)

# --------------------------------
# Helper functions
# --------------------------------
def neighbors4(r, c, n=N):
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        rr, cc = r+dr, c+dc
        if 0 <= rr < n and 0 <= cc < n:
            yield rr, cc

def count_blobs(grid, val):
    n = grid.shape[0]
    vis = np.zeros_like(grid, dtype=bool)
    sizes = []
    for r in range(n):
        for c in range(n):
            if not vis[r,c] and grid[r,c] == val:
                q = deque([(r,c)]); vis[r,c] = True; s = 0
                while q:
                    rr,cc = q.popleft(); s += 1
                    for nr,nc in neighbors4(rr,cc,n):
                        if not vis[nr,nc] and grid[nr,nc] == val:
                            vis[nr,nc] = True; q.append((nr,nc))
                sizes.append(s)
    return sizes

def road_left_to_right_connected(grid):
    n = grid.shape[0]
    vis = np.zeros((n,n), dtype=bool); q = deque()
    for r in range(n):
        if grid[r,0] == ROAD:
            q.append((r,0)); vis[r,0] = True
    while q:
        r,c = q.popleft()
        if c == n-1: return True
        for nr,nc in neighbors4(r,c,n):
            if not vis[nr,nc] and grid[nr,nc] == ROAD:
                vis[nr,nc] = True; q.append((nr,nc))
    return False

def water_top_to_bottom_connected(grid):
    n = grid.shape[0]
    vis = np.zeros((n,n), dtype=bool); q = deque()
    for c in range(n):
        if grid[0,c] == WATER:
            q.append((0,c)); vis[0,c] = True
    while q:
        r,c = q.popleft()
        if r == n-1: return True
        for nr,nc in neighbors4(r,c,n):
            if not vis[nr,nc] and grid[nr,nc] == WATER:
                vis[nr,nc] = True; q.append((nr,nc))
    return False

def enclosed_land_pockets(grid):
    n = grid.shape[0]
    blocked = (grid==ROCK) | (grid==WATER)
    reach = np.zeros((n,n), dtype=bool)
    q = deque()

    # boundary seeds
    for i in range(n):
        for j in (0, n-1):
            if grid[i,j] == LAND and not reach[i,j]:
                reach[i,j] = True; q.append((i,j))
    for j in range(n):
        for i in (0, n-1):
            if grid[i,j] == LAND and not reach[i,j]:
                reach[i,j] = True; q.append((i,j))

    while q:
        r,c = q.popleft()
        for nr,nc in neighbors4(r,c,n):
            if (not reach[nr,nc]) and (not blocked[nr,nc]) and grid[nr,nc] == LAND:
                reach[nr,nc] = True; q.append((nr,nc))

    vis = np.zeros((n,n), dtype=bool); pockets = []
    for r in range(n):
        for c in range(n):
            if grid[r,c] == LAND and not reach[r,c] and not vis[r,c]:
                dq = deque([(r,c)]); vis[r,c] = True; s=0
                while dq:
                    rr,cc = dq.popleft(); s += 1
                    for nr,nc in neighbors4(rr,cc,n):
                        if grid[nr,nc] == LAND and not reach[nr,nc] and not vis[nr,nc]:
                            vis[nr,nc] = True; dq.append((nr,nc))
                pockets.append(s)
    return pockets

# --------------------------------
# Reward Function
# --------------------------------
def compute_chunk_reward(grid, left_col=None, top_row=None, partial=False):
    """
    Reward for a single chunk.
    If partial=True, computes lightweight shaping reward for intermediate steps.
    """
    reward = 0.0
    n = grid.shape[0]

    # baseline structure: roads and rivers
    if road_left_to_right_connected(grid):
        reward += 0.2
    if water_top_to_bottom_connected(grid):
        reward += 0.2

    # lakes & speckles
    w_sizes = count_blobs(grid, WATER)
    if len(w_sizes)>0 and max(w_sizes) >= 8 and len(w_sizes) <= 3:
        reward += 0.2
    reward -= 0.2 * sum(1 for s in w_sizes if s == 1)

    # compact land
    land_sizes = count_blobs(grid, LAND)
    if land_sizes and max(land_sizes) >= 25:
        reward += 0.50

    # cave pockets small bonus
    pockets = enclosed_land_pockets(grid)
    if any(6 <= s <= 60 for s in pockets):
        reward += 0.1

    # diversity penalty (softened with nonlinear scaling)
    counts = np.bincount(grid.flatten(), minlength=NUM_TILES)
    max_ratio = counts.max() / (n*n)
    if max_ratio > 0.70:
        reward -= 2.0 * (max_ratio - 0.70) * 10  # nonlinear scaling
    else:
        if np.count_nonzero(counts) >= 3:
            reward += 0.4

    # continuity with neighbors (roads, water, AND land/rock)
    if left_col is not None:
        this_left = grid[:,0]
        for t, w_match, w_miss in [(ROAD, 0.6, 0.35), (WATER, 0.6, 0.25),
                                   (LAND, 0.3, 0.2), (ROCK, 0.3, 0.2)]:
            match = sum(1 for r in range(n) if left_col[r]==t and this_left[r]==t)
            mismatch = sum(1 for r in range(n) if left_col[r]==t and this_left[r]!=t)
            reward += w_match * match - w_miss * mismatch

    if top_row is not None:
        this_top = grid[0,:]
        for t, w_match, w_miss in [(ROAD, 0.6, 0.35), (WATER, 0.6, 0.25),
                                   (LAND, 0.3, 0.2), (ROCK, 0.3, 0.2)]:
            match = sum(1 for c in range(n) if top_row[c]==t and this_top[c]==t)
            mismatch = sum(1 for c in range(n) if top_row[c]==t and this_top[c]!=t)
            reward += w_match * match - w_miss * mismatch

    # partial shaping: small fraction of final reward
    if partial:
        reward *= 0.05  # scale down so it guides but doesn’t dominate

    return float(reward)

# --------------------------------
# ChunkEnv with Shaping
# --------------------------------
class ChunkEnv:
    def __init__(self, n=N, ci=0, cj=0, total_chunks=1):
        self.n = n
        self.reset()
        self.ci = ci
        self.cj = cj
        self.total_chunks = total_chunks

    def reset(self, left_col=None, top_row=None):
        self.grid = np.full((self.n, self.n), LAND, dtype=int)
        self.t = 0
        self.left_col = left_col
        self.top_row = top_row
        return self.observe()

    def observe(self):
        r, c = divmod(self.t, self.n)
        pos = np.array([r/(self.n-1), c/(self.n-1)], dtype=np.float32)
        placed = self.grid.flatten()[:self.t]
        counts = np.bincount(placed, minlength=NUM_TILES).astype(np.float32)
        counts = counts/self.t if self.t>0 else np.zeros(NUM_TILES, dtype=np.float32)
        up = self.grid[r-1, c] if r-1 >= 0 else -1
        left = self.grid[r, c-1] if c-1 >= 0 else -1
        neigh = np.zeros(NUM_TILES*2, dtype=np.float32)
        if up >= 0: neigh[up] = 1.0
        if left >= 0: neigh[NUM_TILES + left] = 1.0

        # global context: chunk indices normalized
        chunk_pos = np.array([self.ci, self.cj], dtype=np.float32)
        chunk_pos /= np.sqrt(self.total_chunks)  # normalize by grid size

        feat = np.concatenate([pos, counts, neigh, chunk_pos], dtype=np.float32)
        return feat

    def step(self, action:int):
        r, c = divmod(self.t, self.n)
        self.grid[r, c] = int(action)
        self.t += 1
        done = (self.t >= self.n * self.n)

        reward = 0.0
        if done:
            reward = compute_chunk_reward(self.grid, left_col=self.left_col, top_row=self.top_row)
        elif self.t % self.n == 0:
            # shaping reward at each row
            reward = compute_chunk_reward(self.grid, partial=True)

        return self.observe(), reward, done

# --------------------------------
# World-level Environment
# --------------------------------
class MapEnvironment:
    def __init__(self, size=500, chunk_size=10):
        assert size % chunk_size == 0, "size must be divisible by chunk_size"
        self.world_size = size
        self.chunk_size = chunk_size
        self.chunks_per_row = size // chunk_size
        self.num_chunks = self.chunks_per_row * self.chunks_per_row

        self.world = np.full((self.world_size, self.world_size), LAND, dtype=int)
        self.chunks = []
        self.current_chunk_idx = 0
        self.chunk_env = None

        # interface info
        self.state_size = 14 + 2  # previous 14 + chunk pos (ci, cj)
        self.action_size = NUM_TILES

    def _left_top_for_idx(self, idx):
        ci = idx // self.chunks_per_row
        cj = idx % self.chunks_per_row
        left_col = None
        top_row = None
        if cj > 0:
            left_chunk = self.chunks[-1]
            left_col = left_chunk[:, -1].copy()
        if ci > 0:
            top_chunk = self.chunks[(ci - 1) * self.chunks_per_row + cj]
            top_row = top_chunk[-1, :].copy()
        return left_col, top_row, ci, cj

    def reset(self):
        self.world.fill(LAND)
        self.chunks = []
        self.current_chunk_idx = 0
        left_col, top_row, ci, cj = self._left_top_for_idx(0)
        self.chunk_env = ChunkEnv(n=self.chunk_size, ci=ci, cj=cj, total_chunks=self.num_chunks)
        obs = self.chunk_env.reset(left_col=left_col, top_row=top_row)
        return obs

    def step(self, action:int):
        obs, reward, done_chunk = self.chunk_env.step(action)
        if not done_chunk:
            return obs, float(reward), False

        # place chunk
        chunk_grid = self.chunk_env.grid.copy()
        idx = self.current_chunk_idx
        ci = idx // self.chunks_per_row
        cj = idx % self.chunks_per_row
        r0 = ci * self.chunk_size
        c0 = cj * self.chunk_size
        self.world[r0:r0 + self.chunk_size, c0:c0 + self.chunk_size] = chunk_grid
        self.chunks.append(chunk_grid)
        self.current_chunk_idx += 1

        chunk_reward = reward
        if self.current_chunk_idx >= self.num_chunks:
            return None, float(chunk_reward), True

        # new chunk
        left_col, top_row, ci, cj = self._left_top_for_idx(self.current_chunk_idx)
        self.chunk_env = ChunkEnv(n=self.chunk_size, ci=ci, cj=cj, total_chunks=self.num_chunks)
        next_obs = self.chunk_env.reset(left_col=left_col, top_row=top_row)
        return next_obs, float(chunk_reward), False

    def get_full_world(self):
        return self.world.copy()
