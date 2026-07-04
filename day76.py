#Day-76: RL (Reinforce Learning)
import numpy as np
import matplotlib.pyplot as plt

# 1. Build a simple Grid World environment 
class GridWorld:
    def __init__(self):
        self.size   = 4
        self.start  = (0, 0)
        self.goal   = (3, 3)
        self.wall   = (1, 1)   
        self.reset()

    def reset(self):
        self.pos = self.start
        return self.pos   

    def step(self, action):
        r, c = self.pos
        # Apply action — move in chosen direction
        if   action == 0: r = max(0, r-1)          # up
        elif action == 1: r = min(self.size-1, r+1) # down
        elif action == 2: c = max(0, c-1)           # left
        elif action == 3: c = min(self.size-1, c+1) # right

        if (r,c) == self.wall:
            r, c = self.pos   # if would hit wall, stay in place

        self.pos = (r, c)
        done     = self.pos == self.goal
        reward   = 1.0 if done else -0.01  # big reward at goal, small penalty each step
        return self.pos, reward, done

#2. Random Agent (baseline — no learning) 
env  = GridWorld()
n_actions = 4
total_rewards = []

for episode in range(100):
    state       = env.reset()
    total_r     = 0
    for step in range(50):
        action          = np.random.randint(n_actions)  # pick random action
        state, r, done  = env.step(action)
        total_r        += r
        if done: break
    total_rewards.append(total_r)

print(f"Random agent avg reward: {np.mean(total_rewards):.3f}")

# 3. Plot rewards over episodes 
plt.figure(figsize=(8,4))
plt.plot(total_rewards, alpha=0.6, color='steelblue')
plt.axhline(np.mean(total_rewards), color='red', linestyle='--', label=f'Mean={np.mean(total_rewards):.3f}')
plt.xlabel("Episode"); plt.ylabel("Total Reward")
plt.title("Random Agent — Grid World")
plt.legend(); plt.tight_layout()
plt.savefig("rand_agemt.png"); plt.show()

