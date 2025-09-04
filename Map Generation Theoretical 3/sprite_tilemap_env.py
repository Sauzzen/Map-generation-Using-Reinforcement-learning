# sprite_tilemap_env.py
import pygame
import numpy as np
import os
import random
from scipy.ndimage import label
# ------------------------------
# Tile constants (logical types)
# ------------------------------
LAND = 0
WATER = 1
#DESERT = 2
#SNOW = 3

# ------------------------------
# Mapping (row, col) in PNG -> logical tile
# Update according to your TinyRanch_Tiles.png
# ------------------------------
TILE_INDEX_MAP = {
    # Land tiles
    (0,0): LAND, (0,1): LAND, (0,2): LAND, (0,3): LAND,
    (0,4): LAND, (0,5): LAND, (0,6): LAND, (0,7): LAND,
    (1,0): LAND, (1,1): LAND, (1,2): LAND, (1,3): LAND,
    (1,4): LAND, (1,5): LAND, (1,6): LAND, (1,7): LAND,

    # Desert tiles
    #(4,0): DESERT, (4,1): DESERT, (4,2): DESERT, (4,3): DESERT,
    #(5,0): DESERT, (5,1): DESERT, (5,2): DESERT, (5,3): DESERT,

    # Snow tiles
    #(4,4): SNOW, (4,5): SNOW, (4,6): SNOW, (4,7): SNOW,
    #(5,4): SNOW, (5,5): SNOW, (5,6): SNOW, (5,7): SNOW,

    # Water tiles
    (1,8): WATER, (1,9): WATER, (2,8): WATER, (2,9): WATER,
    (3,8): WATER, (3,9): WATER, (4,8): WATER, (4,9): WATER,
    (5,8): WATER, (5,9): WATER,
}

# ------------------------------
# Tile names (for debug)
# ------------------------------
TILE_NAMES = {LAND:"LAND", WATER:"WATER"}

# ------------------------------
# Sprite Tilemap Environment
# ------------------------------
class SpriteTilemapEnv:
    def __init__(self, map_width=16, map_height=16, viewport_width=8, viewport_height=8,
                 sprite_size=50, sprite_sheet="TinyRanch_Tiles.png", sheet_rows=6, sheet_cols=10,
                 debug=False):
        self.map_width = map_width
        self.map_height = map_height
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.sprite_size = sprite_size
        self.debug = debug

        # Full map and variation indices
        self.tilemap = np.full((self.map_height, self.map_width), LAND)
        self.sprite_indices = np.zeros((self.map_height, self.map_width), dtype=int)
        self.current_biome = LAND

        # Viewport offset
        self.offset_x = 0
        self.offset_y = 0

        # PyGame setup
        pygame.init()
        self.screen_width = self.viewport_width * self.sprite_size
        self.screen_height = self.viewport_height * self.sprite_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Sprite Tilemap Debug Environment")
        self.clock = pygame.time.Clock()

        # Load sprites
        self.sprites = self.load_sprites(sprite_sheet, sheet_rows, sheet_cols)

    # ------------------------------
    # Load and slice the PNG
    # ------------------------------
    def load_sprites(self, sprite_sheet_path, sheet_rows, sheet_cols):
        if not os.path.exists(sprite_sheet_path):
            raise FileNotFoundError(f"Cannot find '{sprite_sheet_path}'")
        sheet = pygame.image.load(sprite_sheet_path).convert_alpha()
        sheet_width, sheet_height = sheet.get_size()
        tile_width = sheet_width // sheet_cols
        tile_height = sheet_height // sheet_rows

        sprites = {}
        for (row, col), tile_type in TILE_INDEX_MAP.items():
            if row >= sheet_rows or col >= sheet_cols:
                continue
            x = col * tile_width
            y = row * tile_height
            img = sheet.subsurface((x, y, tile_width, tile_height))
            img = pygame.transform.scale(img, (self.sprite_size, self.sprite_size))
            if tile_type not in sprites:
                sprites[tile_type] = []
            sprites[tile_type].append(img)
        return sprites

    # ------------------------------
    # Reset map with optional biome & size
    # ------------------------------
    def reset(self, biome=None, map_width=None, map_height=None):
        if map_width: self.map_width = map_width
        if map_height: self.map_height = map_height
        self.current_biome = LAND if biome is None else biome
        self.tilemap = np.full((self.map_height, self.map_width), self.current_biome)
        self.sprite_indices = np.zeros((self.map_height, self.map_width), dtype=int)
        self.assign_random_sprites()
        self.offset_x = 0
        self.offset_y = 0
        if self.debug: print(f"[DEBUG] Map reset: biome={TILE_NAMES[self.current_biome]}")
        return self.tilemap

    # ------------------------------
    # Assign random variations to tiles (kept fixed)
    # ------------------------------
    def assign_random_sprites(self):
        for y in range(self.map_height):
            for x in range(self.map_width):
                tile_type = self.tilemap[y, x]
                if tile_type in self.sprites:
                    self.sprite_indices[y, x] = random.randint(0, len(self.sprites[tile_type])-1)

    # ------------------------------
    # Place tile at (x,y)
    # ------------------------------
    def place_sprite(self, x, y, tile_type):
        if 0 <= x < self.map_width and 0 <= y < self.map_height:
            if tile_type == WATER:
                self.tilemap[y, x] = WATER
                self.sprite_indices[y, x] = random.randint(0, len(self.sprites[WATER])-1)
                if y+1 < self.map_height:
                    self.tilemap[y+1, x] = WATER
                    self.sprite_indices[y+1, x] = random.randint(0, len(self.sprites[WATER])-1)
            else:
                if self.tilemap[y, x] == self.current_biome:
                    self.tilemap[y, x] = tile_type
                    self.sprite_indices[y, x] = random.randint(0, len(self.sprites[tile_type])-1)
            if self.debug: print(f"[DEBUG] Placed {TILE_NAMES[tile_type]} at ({x},{y})")
            return True
        if self.debug: print(f"[DEBUG] Invalid placement ({x},{y})")
        return False

    # ------------------------------
    # Compute reward for RL
    # ------------------------------


    def compute_reward(self, x, y, tile_type):
        reward = 0.0

        # ----------------------
        # REGION RULES
        # ----------------------
        if x < self.map_width // 2:
            if tile_type == LAND:
                reward += 1.0
            else:
                reward -= 0.5  # softened penalty for flexibility

        # ----------------------
        # NEIGHBOR CLUSTERING BONUS
        # ----------------------
        neighbors = [(0,1), (0,-1), (1,0), (-1,0)]
        same_count = 0  

        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                if self.tilemap[ny, nx] == tile_type:
                    same_count += 1

        if tile_type == WATER:
            if same_count == 0:
                reward -= 1.5
            elif same_count == 1:
                reward += 0.2
            elif same_count >= 2:
                reward += 0.8  # encourage lakes/rivers

        elif tile_type == LAND:
            reward += same_count * 0.4
            if same_count == 0:
                reward -= 0.8

        # ----------------------
        # GLOBAL BALANCE CHECK
        # ----------------------
        land_count = np.sum(self.tilemap == LAND)
        water_count = np.sum(self.tilemap == WATER)

        total = land_count + water_count
        if total > 0:
            water_ratio = water_count / total
            target_ratio = 0.3  # ~30% water
            reward -= abs(water_ratio - target_ratio) * 1.0  # softer weight

        # ----------------------
        # CONNECTIVITY CHECK
        # ----------------------
        if water_count > 0:
            water_mask = (self.tilemap == WATER)
            _, num_water_regions = label(water_mask)

            # Fewer regions is better
            reward -= (num_water_regions - 1) * 0.5  
            # e.g. 1 lake = no penalty, 3 lakes = -1.0 penalty

        return reward



    # ------------------------------
    # Render viewport
    # ------------------------------
    def render(self):
        for y in range(self.viewport_height):
            for x in range(self.viewport_width):
                map_x = x + self.offset_x
                map_y = y + self.offset_y
                if map_x < self.map_width and map_y < self.map_height:
                    tile_type = self.tilemap[map_y, map_x]
                    if tile_type in self.sprites:
                        img_idx = self.sprite_indices[map_y, map_x]
                        img = self.sprites[tile_type][img_idx]
                        self.screen.blit(img, (x*self.sprite_size, y*self.sprite_size))
                    #pygame.draw.rect(self.screen, (0,0,0),
                    #                 pygame.Rect(x*self.sprite_size, y*self.sprite_size,
                    #                             self.sprite_size, self.sprite_size), 1)
                    if self.debug:
                        font = pygame.font.SysFont(None, 20)
                        text = font.render(str(tile_type), True, (255,255,255))
                        self.screen.blit(text, (x*self.sprite_size+5, y*self.sprite_size+5))
        pygame.display.flip()

    # ------------------------------
    # Scroll viewport
    # ------------------------------
    def scroll(self, dx, dy):
        self.offset_x = max(0, min(self.offset_x+dx, self.map_width-self.viewport_width))
        self.offset_y = max(0, min(self.offset_y+dy, self.map_height-self.viewport_height))

    # ------------------------------
    # Close PyGame
    # ------------------------------
    def close(self):
        pygame.quit()

# ------------------------------
# Quick test
# ------------------------------
if __name__=="__main__":
    env = SpriteTilemapEnv(map_width=8, map_height=8, viewport_width=8, viewport_height=8,
                           sprite_size=50, sprite_sheet="TinyRanch_Tiles.png", debug=True)
    env.reset()

    running = True
    while running:
        env.screen.fill((0,0,0))
        env.render()
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                running=False
            elif event.type==pygame.KEYDOWN:
                if event.key==pygame.K_w: env.scroll(0,-1)
                if event.key==pygame.K_s: env.scroll(0,1)
                if event.key==pygame.K_a: env.scroll(-1,0)
                if event.key==pygame.K_d: env.scroll(1,0)
        env.clock.tick(10)
    env.close()