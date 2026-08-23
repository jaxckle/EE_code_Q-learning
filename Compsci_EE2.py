import numpy as np
import gymnasium as gym
import pandas as pd
from dataclasses import dataclass
import time


class CustomEnv(gym.Env):
    def __init__(self,edge_len=10):
        super().__init__()
        self.edge_len = edge_len
        self.action_space = gym.spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = gym.spaces.Discrete(edge_len**2)
        self.start_pos = (0, 0)
        self.goal_pos = (edge_len - 1, edge_len - 1)

    def flatten_state(self, pos):
        return pos[0] + pos[1] * self.edge_len

    def reset(self, seed=None, options=None):
        # 1. Background Gymnasium reset tracking (highly recommended by the API)
        super().reset(seed=seed)

        self.current_pos = self.start_pos
        self.step_count = 0
        info ={}
        self.intial_state = 0
        return self.flatten_state(self.current_pos), info
    
    def distance_function(self, target_pos, goal_pos):
        reward_distance = float(np.linalg.norm(np.array(target_pos) - np.array(goal_pos)))
        return reward_distance
        
    def step(self, action):
        # current posistion
        row, col = self.current_pos
        self.step_count += 1
        self.target_pos = (row,col)
        if self.action_space.contains(action):
            if action == 0: # down
                self.target_pos = (max(row - 1, 0), col)
            elif action == 1:  # Up
                self.target_pos = (min(row + 1, self.edge_len - 1), col)
            elif action == 2:  # Left
                self.target_pos = (row, max(col - 1, 0))
            elif action == 3:  # Right
                self.target_pos = (row, min(col + 1, self.edge_len - 1))
        # dynamic walls
        static_wall = [(0,2), (0,3), (0,4), (0,5), (0,6),(0,7),
                        (3,2), (3,3), (3,4), (3,5), (3,6), (3,7),
                        (6,2), (6,3), (6,4), (6,5), (6,6), (6,7),
                        (9,2), (9,3), (9,4), (9,5), (9,6), (9,7)]
        
        change_wall = [(8-self.step_count%3, 0), (1, 2+self.step_count%3),
                       (2, 5+self.step_count%3), (4, 2+self.step_count%3), 
                       (5, 5+self.step_count%3), (7, 2+self.step_count%3),
                       (8, 5+self.step_count%3),(0,8+self.step_count%3),
                       (6,8+self.step_count%3)]
        collision_log = []
        obsticals = static_wall + change_wall
        # collision detection
        terminated = False
        collision = False

        if self.target_pos in static_wall:
            collision = True
            reward = -1.0  
            collision_log.append(self.target_pos)
            
        elif self.target_pos in change_wall:
            collision = True
            reward = -self.distance_function(self.target_pos, self.goal_pos)
            collision_log.append(self.target_pos)
        else:
            self.current_pos = self.target_pos
            collision = False
            
            if self.current_pos == self.goal_pos:
                reward = 10.0
                terminated = True
            else:
                reward = -0.05  
        # 1D q table
        flattened_state = self.flatten_state(self.current_pos)
        info = {
            "current_pos": self.current_pos,
            "target_pos": self.target_pos,
            "obsticals": obsticals,
            "collision_log": collision_log,
            "collision": collision,
            "step_count": self.step_count,
            "exact_distance": float(self.distance_function(self.current_pos, self.goal_pos)),
            "exact_position": self.current_pos,

        }

        return flattened_state, reward, terminated, False, info

@dataclass
class Settings:
    total_episodes: int = 3000 
    max_steps_per_episode: int = 100  
    learning_rate: float = 0.5   
    gamma: float = 0.99           
    epsilon_start: float = 0.99   
    epsilon_min: float = 0.02    
    epsilon_decay: float = 0.02   
    seed: int = 42
    n_runs: int = 100
    edge_len: int = 10

class Qlearning:
    def __init__(self, learning_rate, gamma, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reset_qtable()

    def update(self, state, action, reward, new_state, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.qtable[new_state, :])

        delta = target - self.qtable[state, action]
        q_update = self.qtable[state, action] + self.learning_rate * delta
        return q_update, delta

    def reset_qtable(self):
        """Reset the Q-table."""
        self.qtable = np.zeros((self.state_size, self.action_size))

class EpsilonGreedy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def reset(self):
        self.epsilon = settings.epsilon_start

    def choose_action(self, action_space, state, qtable):
        """Choose an action `a` in the current world state (s)."""
        # First we randomize a number
        explor_exploit_tradeoff = np.random.rand()

        if explor_exploit_tradeoff < self.epsilon:
            action = action_space.sample()

        else:
            max_ids = np.where(qtable[state, :] == max(qtable[state, :]))[0]
            action = np.random.choice(max_ids)
        return action
    def decay_epsilon(self, epsilon_min, epsilon_decay):
        if self.epsilon > epsilon_min:
            self.epsilon = self.epsilon * (1 - epsilon_decay)
        else:
            self.epsilon = epsilon_min
        return self.epsilon

class Sarsa:
    def __init__(self, learning_rate, gamma, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reset_qtable()

    def update(self, state, action, reward, new_state, new_action, done):
        """SARSA update: Q(s,a) := Q(s,a) + lr [target - Q(s,a)]"""
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.qtable[new_state, new_action]

        delta = target - self.qtable[state, action]
        q_update = self.qtable[state, action] + self.learning_rate * delta
        return q_update, delta

    def reset_qtable(self):
        """Reset the Q-table."""
        self.qtable = np.zeros((self.state_size, self.action_size))

def train_q_learning(settings: Settings):
    
    env = CustomEnv(edge_len=settings.edge_len)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    rewards = np.zeros((settings.total_episodes, settings.n_runs))
    steps = np.zeros((settings.total_episodes, settings.n_runs))
    td_errors = np.zeros((settings.total_episodes, settings.n_runs))
    collisions = np.zeros((settings.total_episodes, settings.n_runs))
    final_distance = np.zeros((settings.total_episodes, settings.n_runs))
    times = np.zeros((settings.total_episodes, settings.n_runs))
    qtable_deltas = np.zeros((settings.total_episodes, settings.n_runs))
    paths = np.zeros((settings.total_episodes, settings.n_runs))

    collision_log = np.empty((settings.total_episodes, settings.n_runs), dtype=object)

    learner = Qlearning(settings.learning_rate, settings.gamma, n_states, n_actions)
    explorer = EpsilonGreedy(settings.epsilon_start)


    for run in range(settings.n_runs):
        learner.reset_qtable()
        explorer.reset()

        for episode in range(settings.total_episodes):
            state, _ = env.reset(seed=settings.seed + run)
            total_reward = 0.0
            step = 0
            done = False
            episode_td_errors = []
            episode_collisions = 0
            episode_collision_log = []  # accumulates across the whole episode
            last_distance = None
            old_qtable = learner.qtable.copy()

            start_time = time.perf_counter()
            while not done and step < settings.max_steps_per_episode:
                action = explorer.choose_action(env.action_space, state, learner.qtable)
                new_state, reward, terminated, truncated, info = env.step(action)

                new_q, delta = learner.update(state, action, reward, new_state, done=terminated)
                learner.qtable[state, action] = new_q

                episode_td_errors.append(abs(delta))
                episode_collisions += int(info["collision"])

                episode_collision_log.extend(info["collision_log"])
                last_distance = info["exact_distance"]
                state = new_state
                total_reward += reward
                step += 1
                done = terminated or truncated
                path = step/12

            end_time = time.perf_counter()
            qtable_delta = np.linalg.norm(learner.qtable - old_qtable)
            execution_time = end_time - start_time

            explorer.decay_epsilon(settings.epsilon_min, settings.epsilon_decay)
            times[episode, run] = execution_time
            rewards[episode, run] = total_reward
            steps[episode, run] = step
            td_errors[episode, run] = np.mean(episode_td_errors) if episode_td_errors else 0.0
            collisions[episode, run] = episode_collisions
            final_distance[episode, run] = last_distance if last_distance is not None else np.nan
            qtable_deltas[episode,run] = qtable_delta
            paths[episode, run] = step/12
            # assigned once, after the loop, with the FULL episode's log
            collision_log[episode, run] = episode_collision_log

    env.close()
    return {
        "rewards": rewards,
        "steps": steps,
        "td_errors": td_errors,
        "collisions": collisions,
        "final_distance": final_distance,
        "times": times,
        "final_qtables_shape": (n_states, n_actions),
        "qtable_deltas": qtable_deltas,
        "paths": paths,
        "collision_log": collision_log,
    }

def train_sarsa(settings: Settings):
    env = CustomEnv(edge_len=settings.edge_len)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    rewards = np.zeros((settings.total_episodes, settings.n_runs))
    steps = np.zeros((settings.total_episodes, settings.n_runs))
    td_errors = np.zeros((settings.total_episodes, settings.n_runs))
    collisions = np.zeros((settings.total_episodes, settings.n_runs))
    final_distance = np.zeros((settings.total_episodes, settings.n_runs))
    times = np.zeros((settings.total_episodes, settings.n_runs))
    qtable_deltas = np.zeros((settings.total_episodes, settings.n_runs))
    paths = np.zeros((settings.total_episodes, settings.n_runs))
    collision_log = np.empty((settings.total_episodes, settings.n_runs), dtype=object)

    learner = Sarsa(settings.learning_rate, settings.gamma, n_states, n_actions)
    explorer = EpsilonGreedy(settings.epsilon_start)

    for run in range(settings.n_runs):
        learner.reset_qtable()
        explorer.reset()

        for episode in range(settings.total_episodes):
            state, _ = env.reset(seed=settings.seed + run)
            total_reward = 0.0
            step = 0
            terminated = False
            episode_td_errors = []
            episode_collisions = 0
            episode_collision_log = []
            last_distance = None

            action = explorer.choose_action(env.action_space, state, learner.qtable)
            old_qtable = learner.qtable.copy()


            start_time = time.perf_counter()

            while not terminated and step < settings.max_steps_per_episode:
                new_state, reward, terminated, truncated, info = env.step(action)
                new_action = explorer.choose_action(env.action_space, new_state, learner.qtable)

                new_q, delta = learner.update(state, action, reward, new_state, new_action, done=terminated)
                learner.qtable[state, action] = new_q

                episode_td_errors.append(abs(delta))
                episode_collisions += int(info["collision"])
                episode_collision_log.extend(info["collision_log"])
                last_distance = info["exact_distance"]

                state, action = new_state, new_action
                total_reward += reward
                step += 1
                path = step/12

            end_time = time.perf_counter()
            execution_time = end_time - start_time
            qtable_delta = np.linalg.norm(learner.qtable - old_qtable)

            explorer.decay_epsilon(settings.epsilon_min, settings.epsilon_decay)

            times[episode, run] = execution_time
            rewards[episode, run] = total_reward
            steps[episode, run] = step
            td_errors[episode, run] = np.mean(episode_td_errors) if episode_td_errors else 0.0
            collisions[episode, run] = episode_collisions
            final_distance[episode, run] = last_distance if last_distance is not None else np.nan
            qtable_deltas[episode,run] = qtable_delta
            paths[episode, run] = step/12
            collision_log[episode, run] = episode_collision_log

    env.close()
    return {
        "rewards": rewards,
        "steps": steps,
        "td_errors": td_errors,
        "collisions": collisions,
        "final_distance": final_distance,
        "times": times,
        "final_qtables_shape": (n_states, n_actions),
        "qtable_deltas": qtable_deltas,
        "paths": paths,
        "collision_log": collision_log,
    }

def results_to_dataframe(results: dict, algorithm_name: str):
    n_episodes, n_runs = results["rewards"].shape
    episodes = np.tile(np.arange(n_episodes), reps=n_runs)

    df = pd.DataFrame({
        "episode": episodes,
        "reward": results["rewards"].flatten(order="F"),
        "steps": results["steps"].flatten(order="F"),
        "td_error": results["td_errors"].flatten(order="F"),
        "collisions": results["collisions"].flatten(order="F"),
        "final_distance": results["final_distance"].flatten(order="F"),
        "execution_time": results["times"].flatten(order="F"),
        "qtable_deltas": results["qtable_deltas"].flatten(order="F"),
        "paths": results["paths"].flatten(order="F"),

        "collision_log": results["collision_log"].flatten(order="F"),
    })
    df["algorithm"] = algorithm_name
    return df

if __name__ == "__main__":
    settings = Settings()

    print("Training Q-learning...")
    q_results = train_q_learning(settings)

    print("Training SARSA...")
    sarsa_results = train_sarsa(settings)

    q_df = results_to_dataframe(q_results, "Q-learning")
    sarsa_df = results_to_dataframe(sarsa_results, "SARSA")
    combined_df = pd.concat([q_df, sarsa_df], ignore_index=True)

    combined_df.to_csv("training_results.csv", index=False)
