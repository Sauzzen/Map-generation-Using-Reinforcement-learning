# visualize.py
# Save chunk PNGs and stitch into world PNGs

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# color map for tiles
COLORS = {
    0: (141, 217, 109),   # LAND - green
    1: (51, 102, 255),    # WATER - blue
    2: (170, 170, 170),   # ROCK - gray
    3: (30, 30, 30),      # ROAD - dark
}

def _tile_to_rgb(tile):
    return COLORS.get(int(tile), (0, 0, 0))

def save_chunk_png(chunk_grid, path_png, scale=20):
    """
    Save a single chunk (e.g., 10x10 grid) as a PNG.
    """
    n = chunk_grid.shape[0]
    img = Image.new("RGB", (n, n))
    px = img.load()
    for i in range(n):
        for j in range(n):
            px[j, i] = _tile_to_rgb(chunk_grid[i, j])
    # nearest-neighbor resize for crisp tiles
    img = img.resize((n * scale, n * scale), resample=Image.NEAREST)
    os.makedirs(os.path.dirname(path_png), exist_ok=True)
    img.save(path_png)

# Alias for backwards compatibility
save_chunk_image = save_chunk_png

def stitch_chunks(chunks_list, chunks_per_row=5):
    """
    Combine multiple chunks into a full map.
    """
    if len(chunks_list) == 0:
        return None
    chunk_n = chunks_list[0].shape[0]
    rows = []
    per_row = chunks_per_row
    nrows = (len(chunks_list) + per_row - 1) // per_row
    for r in range(nrows):
        start = r * per_row
        row_chunks = chunks_list[start:start + per_row]
        # if last row incomplete, pad with LAND chunks
        if len(row_chunks) < per_row:
            pad = [np.full((chunk_n, chunk_n), 0, dtype=int)] * (per_row - len(row_chunks))
            row_chunks = row_chunks + pad
        row = np.hstack(row_chunks)
        rows.append(row)
    world = np.vstack(rows)
    return world

def save_world_png(world_grid, path_png, scale=10):
    """
    Save a stitched world (numpy 2D array) as PNG.
    """
    h, w = world_grid.shape
    img = Image.new("RGB", (w, h))
    px = img.load()
    for i in range(h):
        for j in range(w):
            px[j, i] = _tile_to_rgb(world_grid[i, j])
    img = img.resize((w * scale, h * scale), resample=Image.NEAREST)
    os.makedirs(os.path.dirname(path_png), exist_ok=True)
    img.save(path_png)

def save_full_map(full_map, episode, save_dir="full_maps", scale=10):
    """
    Save the entire map (50x50 or 100x100) as a PNG image using tile colors.
    """
    os.makedirs(save_dir, exist_ok=True)

    h, w = full_map.shape
    img = Image.new("RGB", (w, h))
    px = img.load()
    for i in range(h):
        for j in range(w):
            px[j, i] = _tile_to_rgb(full_map[i, j])

    img = img.resize((w * scale, h * scale), resample=Image.NEAREST)
    filepath = os.path.join(save_dir, f"full_map_ep{episode}.png")
    img.save(filepath)
