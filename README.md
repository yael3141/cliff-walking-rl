import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from environment import CliffWalkingEnv, GOAL, START  # noqa: E402


class TestCliffWalkingEnv(unittest.TestCase):
    def test_reset_returns_start(self):
        env = CliffWalkingEnv()
        self.assertEqual(env.reset(), START)

    def test_normal_step_costs_one_and_is_not_done(self):
        env = CliffWalkingEnv()
        env.reset()
        result = env.step("up")
        self.assertEqual(result.reward, -1.0)
        self.assertFalse(result.done)
        self.assertEqual(result.state, (2, 0))

    def test_walking_into_cliff_resets_to_start_with_big_penalty(self):
        env = CliffWalkingEnv()
        env.reset()
        result = env.step("right")  # from (3,0) -> (3,1), which is in the cliff
        self.assertEqual(result.reward, -100.0)
        self.assertFalse(result.done)
        self.assertEqual(result.state, START)

    def test_reaching_goal_ends_episode(self):
        env = CliffWalkingEnv()
        env.reset()
        env.step("up")  # (2, 0)
        for _ in range(11):
            env.step("right")  # walk along row 2 to (2, 11)
        result = env.step("down")  # (3, 11) == GOAL
        self.assertTrue(result.done)
        self.assertEqual(result.state, GOAL)

    def test_actions_are_clipped_at_grid_boundary(self):
        env = CliffWalkingEnv()
        env.reset()
        result = env.step("down")  # already in bottom row, should stay in place (not error)
        self.assertEqual(result.state, START)

    def test_state_to_index_is_bijective_over_grid(self):
        seen = set()
        for r in range(4):
            for c in range(12):
                idx = CliffWalkingEnv.state_to_index((r, c))
                self.assertNotIn(idx, seen)
                seen.add(idx)
        self.assertEqual(len(seen), CliffWalkingEnv.n_states())


if __name__ == "__main__":
    unittest.main()
