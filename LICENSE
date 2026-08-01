import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from q_learning import TDAgent  # noqa: E402


class TestTDAgentUpdateRules(unittest.TestCase):
    def test_q_learning_bootstraps_off_greedy_next_action(self):
        """Q-learning's target uses max_a' Q(s', a'), regardless of which
        action a_next is passed in -- that's what makes it off-policy."""
        agent = TDAgent(n_states=3, algorithm="q_learning", alpha=1.0, gamma=1.0)
        agent.Q[1] = np.array([5.0, 2.0, 0.0, -1.0])  # max is 5.0, at action 0

        agent.update(s=0, a=0, r=1.0, s_next=1, a_next=3, done=False)  # a_next=3 (Q=-1) should be ignored
        # target = r + gamma * max(Q[1]) = 1 + 5 = 6; Q[0,0] was 0, alpha=1 -> new value = 6
        self.assertAlmostEqual(agent.Q[0, 0], 6.0)

    def test_sarsa_bootstraps_off_actual_next_action(self):
        """SARSA's target uses Q(s', a_next) for the *specific* a_next
        passed in, even if it isn't the greedy action -- that's what makes
        it on-policy (it accounts for the agent's own exploration)."""
        agent = TDAgent(n_states=3, algorithm="sarsa", alpha=1.0, gamma=1.0)
        agent.Q[1] = np.array([5.0, 2.0, 0.0, -1.0])

        agent.update(s=0, a=0, r=1.0, s_next=1, a_next=3, done=False)  # explicitly the -1.0 action
        # target = r + gamma * Q[1, 3] = 1 + (-1) = 0
        self.assertAlmostEqual(agent.Q[0, 0], 0.0)

    def test_terminal_update_ignores_bootstrap(self):
        agent = TDAgent(n_states=3, algorithm="q_learning", alpha=1.0, gamma=1.0)
        agent.Q[1] = np.array([100.0, 100.0, 100.0, 100.0])  # would dominate if not ignored
        agent.update(s=0, a=0, r=-1.0, s_next=1, a_next=0, done=True)
        self.assertAlmostEqual(agent.Q[0, 0], -1.0)

    def test_epsilon_zero_is_always_greedy(self):
        agent = TDAgent(n_states=2, algorithm="q_learning", epsilon=0.0)
        agent.Q[0] = np.array([1.0, 5.0, 2.0, 0.0])
        for _ in range(20):
            self.assertEqual(agent.epsilon_greedy_action(0), 1)

    def test_unknown_algorithm_raises(self):
        agent = TDAgent(n_states=2, algorithm="not_a_real_algorithm")
        with self.assertRaises(ValueError):
            agent.update(s=0, a=0, r=0.0, s_next=1, a_next=0, done=False)


if __name__ == "__main__":
    unittest.main()
