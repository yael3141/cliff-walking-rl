"""Generates the two key plots: (1) the reward-per-episode learning curve
that reproduces Sutton & Barto Figure 6.4, and (2) a visualization of each
algorithm's final greedy policy over the grid."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from environment import ACTIONS, CLIFF, GOAL, N_COLS, N_ROWS, START, CliffWalkingEnv
from q_learning import train

ACTION_ARROWS = {"up": "↑", "down": "↓", "left": "←", "right": "→"}


def _smooth(x: np.ndarray, window: int = 20) -> np.ndarray:
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot_learning_curves(rewards_q: np.ndarray, rewards_s: np.ndarray, out_path: str) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(_smooth(rewards_q), label="Q-learning (off-policy)", linewidth=1.8)
    plt.plot(_smooth(rewards_s), label="SARSA (on-policy)", linewidth=1.8)
    plt.ylim(-100, 0)
    plt.xlabel("Episode")
    plt.ylabel("Sum of rewards during episode (20-episode moving average)")
    plt.title("Cliff Walking: Q-learning vs. SARSA\n(reproducing Sutton & Barto, Fig. 6.4)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def _rollout_path(agent) -> list[tuple[int, int]]:
    env = CliffWalkingEnv()
    state = env.reset()
    path = [state]
    for _ in range(200):
        s = env.state_to_index(state)
        a = int(agent.Q[s].argmax())
        result = env.step(ACTIONS[a])
        state = result.state
        path.append(state)
        if result.done:
            break
    return path


def plot_policy_grid(agent, title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    grid = np.zeros((N_ROWS, N_COLS))
    for r, c in CLIFF:
        grid[r, c] = -1
    grid[GOAL] = 1
    ax.imshow(grid, cmap="Pastel1", vmin=-1, vmax=1)

    for r in range(N_ROWS):
        for c in range(N_COLS):
            if (r, c) in CLIFF or (r, c) == GOAL:
                continue
            s = CliffWalkingEnv.state_to_index((r, c))
            a = ACTIONS[int(agent.Q[s].argmax())]
            ax.text(c, r, ACTION_ARROWS[a], ha="center", va="center", fontsize=11)

    ax.text(START[1], START[0], "S", ha="center", va="center", fontweight="bold", color="blue")
    ax.text(GOAL[1], GOAL[0], "G", ha="center", va="center", fontweight="bold", color="green")

    path = _rollout_path(agent)
    xs = [c for _, c in path]
    ys = [r for r, _ in path]
    ax.plot(xs, ys, color="crimson", linewidth=2, alpha=0.6)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{title} — greedy policy (path length: {len(path)})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def main() -> None:
    agent_q, rewards_q = train("q_learning", n_episodes=3000, seed=0)
    agent_s, rewards_s = train("sarsa", n_episodes=3000, seed=0)

    plot_learning_curves(rewards_q, rewards_s, "reports/learning_curves.png")
    plot_policy_grid(agent_q, "Q-learning", "reports/policy_q_learning.png")
    plot_policy_grid(agent_s, "SARSA", "reports/policy_sarsa.png")

    import json

    metrics = {
        "n_episodes": 3000,
        "avg_reward_last_100_episodes": {
            "q_learning": float(rewards_q[-100:].mean()),
            "sarsa": float(rewards_s[-100:].mean()),
        },
        "final_greedy_path_length": {
            "q_learning": len(_rollout_path(agent_q)),
            "sarsa": len(_rollout_path(agent_s)),
        },
    }
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
