import numpy as np
from scipy.ndimage import label

class RuleBasedPolisher:
    def __init__(self, max_water_ratio=0.4, min_mountains=5):
        self.max_water_ratio = max_water_ratio
        self.min_mountains = min_mountains

    def polish(self, world):
        world = world.copy()

        # 1. Limit water dominance
        water_ratio = np.mean(world == "water")
        if water_ratio > self.max_water_ratio:
            excess = int((water_ratio - self.max_water_ratio) * world.size)
            water_indices = np.argwhere(world == "water")
            np.random.shuffle(water_indices)
            for idx in water_indices[:excess]:
                world[tuple(idx)] = np.random.choice(["land", "forest"])

        # 2. Ensure minimum mountains
        if np.sum(world == "mountain") < self.min_mountains:
            land_indices = np.argwhere(world == "land")
            np.random.shuffle(land_indices)
            for idx in land_indices[:self.min_mountains]:
                world[tuple(idx)] = "mountain"

        # 3. Smooth isolated tiles
        world = self.smooth_map(world)

        return world

    def smooth_map(self, world):
        new_world = world.copy()
        rows, cols = world.shape
        for r in range(rows):
            for c in range(cols):
                neighbors = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbors.append(world[nr, nc])
                if neighbors:
                    # replace if majority neighbor type is different
                    if neighbors.count(world[r,c]) < len(neighbors)//2:
                        new_world[r,c] = max(set(neighbors), key=neighbors.count)
        return new_world
