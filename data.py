import matplotlib.pyplot as plt
import pandas as pd
import contextlib
import io
from collections import Counter

df = pd.read_csv("training_results.csv")

RL_type = df['algorithm'] == 'Q-learning'

df_qlearning = df[RL_type]
df_SARSA = df[~RL_type]

def average_runs(df, label):
    for column in df.columns:
        if column != 'episode' and column != 'algorithm' and column != 'collision_log':
            avg = df.groupby('episode')[column].mean().reset_index()

            
            plt.figure(column)
            plt.plot(avg['episode'], avg[column], label=label)
            plt.xlabel('Episode')
            plt.ylabel(column)
            plt.title(f'Performance of {column}' if column != 'paths' else 'Optimal Paths Ratio')
            plt.legend()

def total_execution_time(df):
    average_runtime = df['execution_time'].mean()
    print(f"Average execution time: {average_runtime} seconds")
    return average_runtime

def first_solved_episode(df):
    solved = df[df['final_distance'] == 0]
    if solved.empty:
        print("No solved episodes")
        return None
    first_ep = solved['episode'].iloc[0]
    print(f"First solved episode: {first_ep}")
    return first_ep

def number_of_solved_episodes(df):
    solved_episodes = (df['final_distance'] == 0).sum()
    print(f"Number of solved episodes: {solved_episodes}")
    return solved_episodes

def shortest_path_length(df):
    shortest_path = df['steps'].min()
    print(f"Shortest path length: {shortest_path}")
    return shortest_path

def average_path_length(df):
    avg = df['steps'].mean()
    print(f"Average path length: {avg}")
    return avg

def average_reward(df):
    avg = df['reward'].mean()
    print(f"Average reward: {avg}")
    return avg

def max_reward(df):
    result = df['reward'].max()
    print(f"Max reward: {result}")
    return result

def min_reward(df):
    result = df['reward'].min()
    print(f"Min reward: {result}")
    return result

def average_collisions(df):
    avg = df['collisions'].mean()
    print(f"Average collisions: {avg}")
    return avg

def max_collisions(df):
    result = df['collisions'].max()
    print(f"Max collisions: {result}")
    return result

def min_collisions(df):
    result = df['collisions'].min()
    print(f"Min collisions: {result}")
    return result

def reward_before_200(df):
    avg = df[df['episode'] < 200]['reward'].mean()
    print(f"Average reward before episode 200: {avg}")
    return avg

def reward_after_200(df):
    avg = df[df['episode'] >= 200]['reward'].mean()
    print(f"Average reward after episode 200: {avg}")
    return avg

def after_200_steps(df):
    avg = df[df['episode'] >= 200]['steps'].mean()
    print(f"Average steps after episode 200: {avg}")
    return avg

def after_2500_steps(df):
    avg = df[df['episode'] >= 2500]['steps'].mean()
    print(f"Average steps after episode 2500: {avg}")
    return avg

def temporal_difference_variance_after_200(df):
    var = df[df['episode'] >= 200]['td_error'].var()
    print(f"Temporal difference variance after episode 200: {var}")
    return var

def plus_minus_all_stats(df):
    values = {
        "reward_before_200": df[df['episode'] < 200]['reward'].std(),
        "reward_after_200": df[df['episode'] >= 200]['reward'].std(),
        "steps_after_200": df[df['episode'] >= 200]['steps'].std(),
        "collisions_after_200": df[df['episode'] >= 200]['collisions'].std(),
        "TD_error_after_200": df[df['episode'] >= 200]['td_error'].std(),
        "distance_after_200": df[df['episode'] >= 200]['final_distance'].std(),
        "path_ratio_after_200": df[df['episode'] >= 200]['paths'].std(),
    }
    return values

def confidence_interval(values, confidence=0.95):
    mean = values.mean()
    std_err = values.std() / (len(values) ** 0.5)
    margin = std_err * 1.96  # for 95% confidence
    return mean - margin, mean + margin

def coefficient_of_variation(values):
    return values.std() / values.mean() if values.mean() != 0 else 0

def episodes_to_converge(df, threshold=0.05):
    episodes_coverged = df[df['qtable_deltas'] < threshold]
    return len(episodes_coverged) if not episodes_coverged.empty else None

def datatable(df):
    metrics = {

        "Temporal Difference Variance": temporal_difference_variance_after_200(df)
    }
    return metrics

def plot_collision_histogram(df_q, df_s, top_n=10):
    q_counts = dict(collision_location_counts(df_q, top_n))
    s_counts = dict(collision_location_counts(df_s, top_n))
 
    # combine so both bars use the same set/order of locations
    all_locations = sorted(
        set(q_counts) | set(s_counts),
        key=lambda loc: q_counts.get(loc, 0) + s_counts.get(loc, 0),
        reverse=True,
    )[:top_n]
 
    labels = [str(loc) for loc in all_locations]
    q_values = [q_counts.get(loc, 0) for loc in all_locations]
    s_values = [s_counts.get(loc, 0) for loc in all_locations]
 
    x = range(len(all_locations))
    width = 0.4
 
    plt.figure("collision_histogram")
    plt.bar([i - width / 2 for i in x], q_values, width=width, label='Q-learning')
    plt.bar([i + width / 2 for i in x], s_values, width=width, label='SARSA')
    plt.xticks(list(x), labels, rotation=45, ha='right')
    plt.xlabel('Grid square (row, col)')
    plt.ylabel('Number of collisions')
    plt.title(f'Top {top_n} Collision Locations')
    plt.legend()
    plt.tight_layout()
    plt.show()

def count_optimal_paths(df):
    optimal_paths = df[df['steps'] == 18]
    return len(optimal_paths)

def collision_location_counts(df, top_n=10):
    all_collisions = []
    for collision_log in df['collision_log']:
        if pd.notna(collision_log):
            # Convert string representation of list to actual list
            collisions = eval(collision_log)
            all_collisions.extend(collisions)
 
    counter = Counter(all_collisions)
    return counter.most_common(top_n)

def print_full_report(df_q, df_s):
 
    def quiet_call(func, *args):
        """Run func with its internal prints suppressed, return its value."""
        with contextlib.redirect_stdout(io.StringIO()):
            return func(*args)
 
    print("=" * 70)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 70)
    simple_stats = [
        ("Average execution time (s)", total_execution_time),
        ("First solved episode", first_solved_episode),
        ("Number of solved episodes", number_of_solved_episodes),
        ("Shortest path length", shortest_path_length),
        ("Average path length", average_path_length),
        ("Average reward", average_reward),
        ("Max reward", max_reward),
        ("Min reward", min_reward),
        ("Average collisions", average_collisions),
        ("Max collisions", max_collisions),
        ("Min collisions", min_collisions),
        ("Average reward before ep 200", reward_before_200),
        ("Average reward after ep 200", reward_after_200),
        ("Average steps after ep 200", after_200_steps),
        ("Average steps after ep 2500", after_2500_steps),
        ("TD-error variance after ep 200", temporal_difference_variance_after_200),
        ("Count of optimal paths (18 steps)", count_optimal_paths)
    ]
    for label, func in simple_stats:
        q_val = quiet_call(func, df_q)
        s_val = quiet_call(func, df_s)
        print(f"  {label:35s} Q-learning: {q_val!s:<15} SARSA: {s_val}")
 
    print("\n" + "=" * 70)
    print("STANDARD DEVIATIONS (post-episode-200 / 2500 windows)")
    print("=" * 70)
    q_std = plus_minus_all_stats(df_q)
    s_std = plus_minus_all_stats(df_s)
    for key in q_std:
        print(f"  {key:25s} Q-learning: {q_std[key]:<12.4f} SARSA: {s_std[key]:.4f}")
 
    print("\n" + "=" * 70)
    print("CONFIDENCE INTERVALS (95%)")
    print("=" * 70)
    for col in ['reward', 'collisions', 'steps', 'td_error', 'qtable_deltas']:
        q_ci = confidence_interval(df_q[col])
        s_ci = confidence_interval(df_s[col])
        print(f"  [{col:10s}] Q-learning: ({q_ci[0]:.3f}, {q_ci[1]:.3f})   "
              f"SARSA: ({s_ci[0]:.3f}, {s_ci[1]:.3f})")
 
    print("\n" + "=" * 70)
    print("COEFFICIENT OF VARIATION")
    print("=" * 70)
    for col in ['reward', 'collisions', 'steps', 'td_error', 'qtable_deltas']:
        q_cv = coefficient_of_variation(df_q[col])
        s_cv = coefficient_of_variation(df_s[col])
        print(f"  [{col:10s}] Q-learning: {q_cv:.3f}   SARSA: {s_cv:.3f}")
 
    print("\n" + "=" * 70)
    print("CONVERGENCE EPISODE")
    print("=" * 70)
    print(f"  Q-learning: episode {episodes_to_converge(df_q)}")
    print(f"  SARSA:      episode {episodes_to_converge(df_s)}")
 
    print("\n" + "=" * 70)
    print("TOP COLLISION LOCATIONS")
    print("=" * 70)
    print(f"  Q-learning: {collision_location_counts(df_q, top_n=5)}")
    print(f"  SARSA:      {collision_location_counts(df_s, top_n=5)}")
 
    # --- graphs ---
    average_runs(df_q, 'Q-learning')
    average_runs(df_s, 'SARSA')
    plot_collision_histogram(df_q, df_s, top_n=10)
    plt.show()
 
 
print_full_report(df_qlearning, df_SARSA)
