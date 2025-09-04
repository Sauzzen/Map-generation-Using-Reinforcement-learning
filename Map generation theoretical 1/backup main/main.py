# main.py
import os
import numpy as np
from map_env import MapEnvironment
from train_agent import RLAgent
from visualize_map import save_full_map
from rule_based_polish import RuleBasedPolisher

# -----------------
# Training settings
# -----------------
EPISODES = 1500
WORLD_SIZE = 100      # updated to larger world
CHUNK_SIZE = 10
SAVE_INTERVAL = 10
TRAIN_INTERVAL = 4
TARGET_UPDATE = 20
VIS_INTERVAL = 50

# -----------------
# Setup directories
# -----------------
os.makedirs("worlds", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)


def main():
    # Environment + agent
    env = MapEnvironment(size=WORLD_SIZE, chunk_size=CHUNK_SIZE)
    agent = RLAgent(state_size=env.state_size, action_size=env.action_size)

    # Load previous best model if available
    best_model_path = "checkpoints/best_model.h5"
    best_score = -float("inf")
    best_world = None
    if os.path.exists(best_model_path):
        agent.load(best_model_path)
        print("✅ Loaded previous best model.")
        # Optional: load a saved best score
        # best_score = load_saved_score()  

    # Epsilon-greedy settings
    epsilon, epsilon_min, epsilon_decay = 1.0, 0.05, 0.995

    step_count = 0

    for episode in range(1, EPISODES + 1):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state, epsilon)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            step_count += 1

            # Train agent every few steps
            if step_count % TRAIN_INTERVAL == 0 and len(agent.memory) >= agent.batch_size:
                loss = agent.replay()

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Update target network
        if episode % TARGET_UPDATE == 0:
            agent.update_target()

        # Save map visualization
        world = env.get_full_world()
        if episode % VIS_INTERVAL == 0:
            save_full_map(world, episode, save_dir="worlds")

        # Track best performing model
        if total_reward > best_score:
            best_score = total_reward
            best_world = world.copy()
            agent.save(best_model_path)
            print(f"[ep {episode}] 🎉 New best! score={best_score:.2f}, epsilon={epsilon:.3f}")

        # Regular checkpoints
        if episode % SAVE_INTERVAL == 0:
            agent.save(f"checkpoints/ep{episode}.h5")
            print(f"[ep {episode}] ✅ Checkpoint saved; score={total_reward:.2f}")

    # -----------------
    # Final polish + save
    # -----------------
    if best_world is not None:
        polisher = RuleBasedPolisher()
        best_world = polisher.apply(best_world)
        save_full_map(best_world, "final_best", save_dir="worlds")
        print("🏁 Training complete. Best world saved as worlds/final_best_world.png")
    else:
        print("⚠️ No best world generated!")


if __name__ == "__main__":
    main()
