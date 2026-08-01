"""
Tabular Q-Learning (off-policy TD control) and SARSA (on-policy TD control),
implemented from scratch on a shared epsilon-greedy scaffold so the *only*
difference between the two algorithms is the one-line update rule -- which
is exactly where their behavioral difference comes from.

Q-Learning update (off-policy -- bootstraps off the *greedy* next action,
regardless of what the agent actually does next):
    Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

SARSA update (on-policy -- bootstraps off the action the agent *actually*
takes next, including exploration):
    Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from environment import ACTIONS, CliffWalkingEnv


@dataclass
class TDAgent:
    n_states: int
    n_actions: int = len(ACTIONS)
    alpha: float = 0.5
    gamma: float = 1.0
    epsilon: float = 0.1
    algorithm: str = "q_learning"  # "q_learning" or "sarsa"
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    Q: np.ndarray = field(init=False)

    def __post_init__(self):
        self.Q = np.zeros((self.n_states, self.n_actions))

    def epsilon_greedy_action(self, state_idx: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.Q[state_idx]))

    def update(self, s: int, a: int, r: float, s_next: int, a_next: int | None, done: bool) -> None:
        current = self.Q[s, a]
        if done:
            target = r
        elif self.algorithm == "q_learning":
            target = r + self.gamma * np.max(self.Q[s_next])
        elif self.algorithm == "sarsa":
            target = r + self.gamma * self.Q[s_next, a_next]
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        self.Q[s, a] = current + self.alpha * (target - current)

    def greedy_policy(self) -> np.ndarray:
        return np.argmax(self.Q, axis=1)


def train(algorithm: str, n_episodes: int = 3000, seed: int = 0) -> tuple[TDAgent, np.ndarray]:
    env = CliffWalkingEnv()
    agent = TDAgent(n_states=env.n_states(), algorithm=algorithm, rng=np.random.default_rng(seed))

    episode_rewards = np.zeros(n_episodes)

    for ep in range(n_episodes):
        state = env.reset()
        s = env.state_to_index(state)
        a = agent.epsilon_greedy_action(s)
        total_reward = 0.0
        done = False

        while not done:
            result = env.step(ACTIONS[a])
            s_next = env.state_to_index(result.state)
            total_reward += result.reward

            if agent.algorithm == "sarsa" and not result.done:
                a_next = agent.epsilon_greedy_action(s_next)
            else:
                a_next = agent.epsilon_greedy_action(s_next)  # needed for q_learning's next action selection too

            agent.update(s, a, result.reward, s_next, a_next, result.done)

            s, a = s_next, a_next
            done = result.done

        episode_rewards[ep] = total_reward

    return agent, episode_rewards

