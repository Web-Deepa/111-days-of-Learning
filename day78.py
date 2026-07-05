#Day78: DQN 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque   
import random
import gymnasium as gym

#1.Environment: CartPole 

env       = gym.make('CartPole-v1')
n_states  = env.observation_space.shape[0]   
n_actions = env.action_space.n               
print(f"States: {n_states}  Actions: {n_actions}")


#2.Build DQN -the neural network that replaces Q-table
def build_dqn():
    # Input: 4-number state → Output: Q-value for each action
    model = tf.keras.Sequential([
        layers.Input(shape=(n_states,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_actions)  
                ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
    return model
dqn = build_dqn()

# 3. Replay Buffer
replay_buffer = deque(maxlen=2000)   

#4. Hyperparameters
gamma     = 0.95    
eps_min   = 0.01
eps_decay = 0.995   
batch_size = 32

def remember(state, action, reward, next_state, done):
    replay_buffer.append((state, action, reward, next_state, done))

def act(state):
    if np.random.random() < epsilon:
        return env.action_space.sample()      
    q_values = dqn.predict(state[np.newaxis], verbose=0)
    return np.argmax(q_values[0])            

def replay():
    if len(replay_buffer) < batch_size:
        return   

    batch = random.sample(replay_buffer, batch_size)
    for state, action, reward, next_state, done in batch:
        target = reward
        if not done:
            target = reward + gamma * np.max(dqn.predict(next_state[np.newaxis], verbose=0))
        q_vals = dqn.predict(state[np.newaxis], verbose=0)
        q_vals[0][action] = target
        dqn.fit(state[np.newaxis], q_vals, verbose=0)


# 5. Training Loop
episodes        = 50  
rewards_history = []
global epsilon   

for episode in range(episodes):
    state   = env.reset()[0]   
    total_r = 0

    for step in range(500):   
        action = act(state)
        next_state, reward, done, truncated, _ = env.step(action)
        remember(state, action, reward, next_state, done)
        replay()              

        state   = next_state
        total_r += reward
        if done or truncated: break

    epsilon = max(eps_min, epsilon * eps_decay)   
    rewards_history.append(total_r)

    if episode % 10 == 0:
        print(f"Episode {episode:3d} | Reward: {total_r:.0f} | Epsilon: {epsilon:.2f}")

env.close()

#6. Plot training progress
plt.figure(figsize=(8,4))
plt.plot(rewards_history, alpha=0.5, color='steelblue', label='Reward')
plt.plot(np.convolve(rewards_history, np.ones(10)/10, mode='valid'),
         color='red', label='Moving avg (10)')
plt.xlabel("Episode"); plt.ylabel("Total Reward")
plt.title("DQN — CartPole")
plt.legend(); plt.tight_layout()
plt.savefig("day78_dqn.png"); plt.show()
