# main.py
import os
import numpy as np
from map_env import MapEnvironment
from train_agent import RLAgent
from visualize_map import save_chunk_png, save_full_map
from rule_based_polish import RuleBasedPolisher

# Settings
EPISODES = 1500
WORLD_SIZE = 50   # final world will be 2500x2500
CHUNK_SIZE = 10   # each chunk is 50x50
SAVE_INTERVAL = 10

# Output directories
os.makedirs("chunks", exist_ok=True)
os.makedirs("worlds", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

def main():
    env = MapEnvironment(size=WORLD_SIZE, chunk_size=CHUNK_SIZE)
    agent = RLAgent(state_size=env.state_size, action_size=env.action_size)

    best_score = -float("inf")
    best_world = None

    for episode in range(1, EPISODES + 1):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay()

        # Save chunks and world
        world = env.get_full_world()
        for i, chunk in enumerate(env.chunks):
            save_chunk_png(chunk, f"chunks/ep{episode}_chunk{i}.png")
        save_full_map(world, episode, save_dir="worlds")

        # Track best score
        if total_reward > best_score:
            best_score = total_reward
            best_world = world
            agent.save("checkpoints/best_model.h5")
            print(f"[ep {episode}] Saved new best model with score={best_score:.2f}")

        if episode % SAVE_INTERVAL == 0:
            agent.save(f"checkpoints/ep{episode}.h5")
            print(f"[ep {episode}] Checkpoint saved; score={total_reward:.2f}")
        
    # Save final best world
    save_full_map(best_world, "final_best", save_dir="worlds")
    print("Training complete. Best world saved as worlds/final_best_world.png")

if __name__ == "__main__":
    main()
