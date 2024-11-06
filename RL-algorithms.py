# Create a 100x100 grid with obstacles in between 2 random points.
# Build an MDP based RL agent to optimise both policies and actions at every state. 
# Benchmark DP method with other RL solutions for the same problem.




import numpy as np
import random
import time
import matplotlib.pyplot as plt

# Create a 100x100 grid with obstacles in between 2 random points.
def create_grid(size=10, obstacle_density=0.2):
    grid = np.zeros((size, size))
    obstacles = np.random.choice(size * size, int(size * size * obstacle_density), replace=False)
    grid[np.unravel_index(obstacles, grid.shape)] = 1  
    return grid

def generate_start_goal(grid):
    empty_positions = np.argwhere(grid == 0)
    start, goal = empty_positions[np.random.choice(len(empty_positions), 2, replace=False)]
    return tuple(start), tuple(goal)

def get_reward(state, goal):
    return 100 if state == goal else -1

# actions at every state
actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
def get_valid_actions(state, grid):
    valid_actions = []
    for a in actions:
        next_state = (state[0] + a[0], state[1] + a[1])
        if 0 <= next_state[0] < len(grid) and 0 <= next_state[1] < len(grid) and grid[next_state] == 0:
            valid_actions.append(a)
    return valid_actions

def q_learning(grid, start, goal, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    q_table = {}
    for _ in range(episodes):
        state = start
        while state != goal:
            if state not in q_table:
                q_table[state] = {a: 0 for a in get_valid_actions(state, grid)}
                
            if random.random() < epsilon:
                action = random.choice(list(q_table[state].keys()))
            else:
                action = max(q_table[state], key=q_table[state].get)
                
            next_state = (state[0] + action[0], state[1] + action[1])
            reward = get_reward(next_state, goal)
            
            if next_state in q_table:
                next_max = max(q_table[next_state].values())
            else:
                next_max = 0
                q_table[next_state] = {a: 0 for a in get_valid_actions(next_state, grid)}
                
            q_table[state][action] = (1 - alpha) * q_table[state][action] + alpha * (reward + gamma * next_max)
            state = next_state
    return q_table

def sarsa(grid, start, goal, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    q_table = {}
    for _ in range(episodes):
        state = start
        if state not in q_table:
            q_table[state] = {a: 0 for a in get_valid_actions(state, grid)}
        
        if random.random() < epsilon:
            action = random.choice(get_valid_actions(state, grid))
        else:
            action = max(q_table[state], key=q_table[state].get)
        
        while state != goal:
            next_state = (state[0] + action[0], state[1] + action[1])
            reward = get_reward(next_state, goal)
            
            if next_state not in q_table:
                q_table[next_state] = {a: 0 for a in get_valid_actions(next_state, grid)}
                
            if random.random() < epsilon:
                next_action = random.choice(get_valid_actions(next_state, grid))
            else:
                next_action = max(q_table[next_state], key=q_table[next_state].get)
                
            q_table[state][action] = (1 - alpha) * q_table[state][action] + alpha * (reward + gamma * q_table[next_state][next_action])
            state, action = next_state, next_action
    return q_table

def monte_carlo(grid, start, goal, episodes=500, gamma=0.9, epsilon=0.1):
    q_table, returns = {}, {}
    for _ in range(episodes):
        state, episode = start, []
        
        while state != goal:
            if state not in q_table:
                q_table[state] = {a: 0 for a in get_valid_actions(state, grid)}
                
            if random.random() < epsilon:
                action = random.choice(list(q_table[state].keys()))
            else:
                action = max(q_table[state], key=q_table[state].get)
                
            next_state = (state[0] + action[0], state[1] + action[1])
            reward = get_reward(next_state, goal)
            episode.append((state, action, reward))
            state = next_state
        
        G = 0
        for state, action, reward in reversed(episode):
            G = reward + gamma * G
            if (state, action) not in returns:
                returns[(state, action)] = []
            returns[(state, action)].append(G)
            q_table[state][action] = np.mean(returns[(state, action)])
    return q_table

def get_path(grid, start, goal, policy):
    path, state = [start], start
    while state != goal and state in policy:
        action = max(policy[state], key=policy[state].get)
        state = (state[0] + action[0], state[1] + action[1])
        path.append(state)
    return path

def visualize_paths(grid, q_path, sarsa_path, mc_path, start, goal):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Q-learning Path", "SARSA Path", "Monte Carlo Path"]
    paths = [q_path, sarsa_path, mc_path]

    for ax, path, title in zip(axs, paths, titles):
        ax.imshow(grid, cmap="gray_r")
        ax.plot(start[1], start[0], 'go', label="Start")
        ax.plot(goal[1], goal[0], 'ro', label="Goal")
        if path:
            path = np.array(path)
            ax.plot(path[:, 1], path[:, 0], 'b-', label="Path")
        ax.set_title(title)
        ax.legend()

    plt.show()

def compare_algorithms(grid_size=10, obstacle_density=0.2, episodes=500):
    grid = create_grid(grid_size, obstacle_density)
    start, goal = generate_start_goal(grid)

    print("Q-learning")
    q_learning_policy = q_learning(grid, start, goal, episodes)
    q_learning_path = get_path(grid, start, goal, q_learning_policy)

    print("SARSA")
    sarsa_policy = sarsa(grid, start, goal, episodes)
    sarsa_path = get_path(grid, start, goal, sarsa_policy)

    print("Monte Carlo")
    monte_carlo_policy = monte_carlo(grid, start, goal, episodes)
    monte_carlo_path = get_path(grid, start, goal, monte_carlo_policy)

    visualize_paths(grid, q_learning_path, sarsa_path, monte_carlo_path, start, goal)
    

# compare_algorithms(grid_size=10, obstacle_density=0.2, episodes=500)

compare_algorithms(grid_size=5, obstacle_density=0.1, episodes=100)  
