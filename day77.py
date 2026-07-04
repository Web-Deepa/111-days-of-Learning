# Day-77:Q-Learning 
import numpy as np
import matplotlib.pyplot as plt

#1.Grid  Sample grid
class GridWorld:
    def __init__(self):
        self.size, self.goal, self.wall = 4, (3,3), (1,1)
        self.reset()
    def reset(self):
        self.pos = (0,0); return self.pos
    def step(self, action):
        r, c = self.pos
        if   action==0: r=max(0,r-1)
        elif action==1: r=min(3,r+1)
        elif action==2: c=max(0,c-1)
        elif action==3: c=min(3,c+1)
        if (r,c)==self.wall: r,c=self.pos
        self.pos=(r,c)
        done   = self.pos==self.goal
        reward = 1.0 if done else -0.01
        return self.pos, reward, done

env       = GridWorld()
n_states  = 4*4    
n_actions = 4
Q         = np.zeros((n_states, n_actions))   

#2. Hyperparameters
alpha   = 0.1    # learning rate 
gamma   = 0.9    # discount factor 
epsilon = 1.0    # exploration rate 
eps_min = 0.01
eps_decay = 0.99  

def state_to_idx(state):
    return state[0] * 4 + state[1]   # convert (row,col) to single index

rewards_history = []

for episode in range(500):
    state   = env.reset()
    total_r = 0

    for _ in range(100):
        s_idx = state_to_idx(state)

        # Epsilon-greedy
        if np.random.random() < epsilon:
            action = np.random.randint(n_actions)   
        else:
            action = np.argmax(Q[s_idx])           

        next_state, reward, done = env.step(action)
        ns_idx = state_to_idx(next_state)

        # Bellman equation: update Q value 
        # Q(s,a) = Q(s,a) + alpha * [reward + gamma*max(Q(s')) - Q(s,a)]
        Q[s_idx, action] += alpha * (reward + gamma * np.max(Q[ns_idx]) - Q[s_idx, action])

        state   = next_state
        total_r += reward
        if done: break

    epsilon = max(eps_min, epsilon * eps_decay)   
    rewards_history.append(total_r)

print(f"Final avg reward (last 50 eps): {np.mean(rewards_history[-50:]):.3f}")
print("\nLearned Q-table (best action per cell):")
actions = ['↑','↓','←','→']
for r in range(4):
    row = ''
    for c in range(4):
        best = actions[np.argmax(Q[r*4+c])]
        row += f" {best} "
    print(row)

plt.figure(figsize=(8,4))
plt.plot(rewards_history, alpha=0.4, color='steelblue', label='Reward per episode')
plt.plot(np.convolve(rewards_history, np.ones(20)/20, mode='valid'),
         color='red', label='Moving average (20)')
plt.xlabel("Episode"); plt.ylabel("Total Reward")
plt.title("Q-Learning — Grid World")
plt.legend(); plt.tight_layout()
plt.savefig("day77_qlearning.png"); plt.show()

