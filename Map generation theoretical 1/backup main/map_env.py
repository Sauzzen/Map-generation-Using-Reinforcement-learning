import numpy as np
from collections import deque

# Tile codes
LAND, WATER = 0, 1
NUM_TILES = 2

# ------------------------------
# Utility functions
# ------------------------------
def neighbors4(r, c, n):
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

def water_top_to_bottom_connected(grid):
    n = grid.shape[0]
    vis = np.zeros((n,n), dtype=bool)
    q = deque()
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
    blocked = (grid == WATER)
    reach = np.zeros((n, n), dtype=bool)
    q = deque()

    # boundary seeds
    for i in range(n):
        for j in (0, n - 1):
            if grid[i, j] == LAND and not reach[i, j]:
                reach[i, j] = True; q.append((i, j))
    for j in range(n):
        for i in (0, n - 1):
            if grid[i, j] == LAND and not reach[i, j]:
                reach[i, j] = True; q.append((i, j))

    # BFS to mark reachable land from boundary
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c, n):
            if (not reach[nr, nc]) and (not blocked[nr, nc]) and grid[nr, nc] == LAND:
                reach[nr, nc] = True; q.append((nr, nc))

    # find enclosed pockets
    vis = np.zeros((n, n), dtype=bool); pockets = []
    for r in range(n):
        for c in range(n):
            if grid[r, c] == LAND and not reach[r, c] and not vis[r, c]:
                dq = deque([(r,c)]); vis[r,c] = True; s = 0
                while dq:
                    rr, cc = dq.popleft(); s += 1
                    for nr, nc in neighbors4(rr, cc, n):
                        if grid[nr, nc] == LAND and not reach[nr, nc] and not vis[nr, nc]:
                            vis[nr, nc] = True; dq.append((nr, nc))
                pockets.append(s)
    return pockets

# ------------------------------
# Reward functions
# ------------------------------
def compute_chunk_reward(grid, left_col=None, top_row=None):
    """
    Compute reward for a chunk. Normalized by chunk size so larger chunks don't explode rewards.
    """
    reward = 0.0
    n = grid.shape[0]  # chunk size
    max_cells = n * n

    # --- River connectivity ---
    if water_top_to_bottom_connected(grid):
        reward += 0.2

    # --- Lake sizes ---
    water_blobs = count_blobs(grid, WATER)
    for w in water_blobs:
        w_frac = w / max_cells  # normalize
        if 0.04 <= w_frac <= 0.1:    # ~4%-10% of chunk
            reward += 0.2
        elif w == 1:
            reward -= 0.2

    # --- Land masses / islands ---
    land_blobs = count_blobs(grid, LAND)
    for l in land_blobs:
        l_frac = l / max_cells
        if l_frac >= 0.25:   # 25%+
            reward += 0.3
        elif l_frac <= 0.05: # <=5%
            reward -= 0.2

    # --- Enclosed pockets ---
    pockets = enclosed_land_pockets(grid)
    for s in pockets:
        s_frac = s / max_cells
        if 0.06 <= s_frac <= 0.6:  # 6%-60% of chunk
            reward += 0.1

    # --- Diversity ---
    if np.count_nonzero(np.bincount(grid.flatten(), minlength=NUM_TILES)) >= 2:
        reward += 0.2

    # --- Continuity with neighbors ---
    if left_col is not None:
        match_water = sum(1 for r in range(n) if left_col[r]==WATER and grid[r,0]==WATER)
        reward += 0.5 * match_water / n
    if top_row is not None:
        match_water = sum(1 for c in range(n) if top_row[c]==WATER and grid[0,c]==WATER)
        reward += 0.5 * match_water / n

    # --- Optional: water fraction constraint ---
    water_frac = np.sum(grid==WATER) / max_cells
    if water_frac < 0.2 or water_frac > 0.5:
        reward -= 0.2

    # --- Normalize total reward ---
    reward /= 2.0  # scaling factor to keep rewards moderate
    return float(reward)

def compute_overlap_reward(current_chunk, left_chunk=None, top_chunk=None, border=2):
    reward = 0.0
    n = current_chunk.shape[0]

    if left_chunk is not None:
        overlap_left = current_chunk[:, :border]; overlap_right = left_chunk[:, -border:]
        reward += np.sum(overlap_left == overlap_right) * 0.3
        reward -= np.sum(overlap_left != overlap_right) * 0.6

    if top_chunk is not None:
        overlap_top = current_chunk[:border, :]; overlap_bottom = top_chunk[-border:, :]
        reward += np.sum(overlap_top == overlap_bottom) * 0.3
        reward -= np.sum(overlap_top != overlap_bottom) * 0.6

    return float(reward)

# ------------------------------
# Environments
# ------------------------------
class ChunkEnv:
    def __init__(self, n):
        self.n = n
        self.reset()

    def reset(self, left_col=None, top_row=None):
        self.grid = np.full((self.n, self.n), LAND, dtype=int)
        self.t = 0
        self.left_col = left_col
        self.top_row = top_row
        return self.observe()

    def observe(self):
        return self.grid.astype(np.float32) / (NUM_TILES-1)

    def step(self, action:int):
        r, c = divmod(self.t, self.n)
        self.grid[r, c] = int(action)
        self.t += 1
        done = (self.t >= self.n * self.n)
        reward = 0.0
        if done:
            reward = compute_chunk_reward(self.grid, left_col=self.left_col, top_row=self.top_row)
        return self.observe(), reward, done

class MapEnvironment:
    def __init__(self, size=50, chunk_size=10):
        assert size % chunk_size == 0, "size must be divisible by chunk_size"
        self.world_size = size
        self.chunk_size = chunk_size
        self.chunks_per_row = size // chunk_size
        self.num_chunks = self.chunks_per_row * self.chunks_per_row

        self.world = np.full((self.world_size, self.world_size), LAND, dtype=int)
        self.chunks = []
        self.current_chunk_idx = 0
        self.chunk_env = None

        self.state_size = (chunk_size, chunk_size)
        self.action_size = NUM_TILES

    def _left_top_for_idx(self, idx):
        ci = idx // self.chunks_per_row
        cj = idx % self.chunks_per_row
        left_col, top_row = None, None
        if cj > 0: left_col = self.chunks[-1][:, -1].copy()
        if ci > 0: top_row = self.chunks[(ci-1)*self.chunks_per_row + cj][-1, :].copy()
        return left_col, top_row

    def reset(self):
        self.world.fill(LAND)
        self.chunks = []
        self.current_chunk_idx = 0
        left_col, top_row = self._left_top_for_idx(0)
        self.chunk_env = ChunkEnv(n=self.chunk_size)
        obs = self.chunk_env.reset(left_col=left_col, top_row=top_row)
        return obs

    def step(self, action:int):
        obs, reward, done_chunk = self.chunk_env.step(action)
        if not done_chunk: return obs, 0.0, False

        # store finished chunk
        chunk_grid = self.chunk_env.grid.copy()
        idx = self.current_chunk_idx
        ci = idx // self.chunks_per_row
        cj = idx % self.chunks_per_row
        r0 = ci * self.chunk_size
        c0 = cj * self.chunk_size
        self.world[r0:r0+self.chunk_size, c0:c0+self.chunk_size] = chunk_grid
        self.chunks.append(chunk_grid)
        self.current_chunk_idx += 1

        # combine reward
        left_chunk = self.chunks[-2] if cj > 0 else None
        top_chunk = self.chunks[(ci-1)*self.chunks_per_row + cj] if ci > 0 else None
        overlap_reward = compute_overlap_reward(chunk_grid, left_chunk, top_chunk, border=2)
        chunk_reward = reward + overlap_reward

        if self.current_chunk_idx >= self.num_chunks:
            return None, float(chunk_reward), True

        # prepare next chunk
        left_col, top_row = self._left_top_for_idx(self.current_chunk_idx)
        self.chunk_env = ChunkEnv(n=self.chunk_size)
        next_obs = self.chunk_env.reset(left_col=left_col, top_row=top_row)
        return next_obs, float(chunk_reward), False

    def get_full_world(self):
        return self.world.copy()
