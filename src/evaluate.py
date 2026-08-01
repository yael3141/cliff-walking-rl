"""
The Cliff Walking environment, from Sutton & Barto, "Reinforcement
Learning: An Introduction" (2nd ed.), Example 6.6.

A 4x12 grid. The agent starts at the bottom-left and must reach the
bottom-right goal. The entire bottom row between them (except the two
endpoints) is a "cliff": stepping there gives a large negative reward and
sends the agent back to the start. Every other step costs -1 (so the agent
is incentivized to reach the goal quickly), and reaching the goal ends the
episode with reward 0.

This is deliberately implemented from scratch (not via gymnasium) with a
gym-like reset()/step() interface, so the environment mechanics are fully
transparent and testable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_ROWS = 4
N_COLS = 12
START = (3, 0)
GOAL = (3, 11)
CLIFF = {(3, c) for c in range(1, 11)}

ACTIONS = ["up", "down", "left", "right"]
ACTION_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


@dataclass
class StepResult:
    state: tuple[int, int]
    reward: float
    done: bool


class CliffWalkingEnv:
    def __init__(self):
        self.state = START

    def reset(self) -> tuple[int, int]:
        self.state = START
        return self.state

    def step(self, action: str) -> StepResult:
        if action not in ACTIONS:
            raise ValueError(f"Unknown action: {action}")

        dr, dc = ACTION_DELTAS[action]
        r, c = self.state
        new_r = min(max(r + dr, 0), N_ROWS - 1)
        new_c = min(max(c + dc, 0), N_COLS - 1)
        new_state = (new_r, new_c)

        if new_state in CLIFF:
            self.state = START
            return StepResult(state=self.state, reward=-100.0, done=False)

        self.state = new_state
        if new_state == GOAL:
            return StepResult(state=new_state, reward=-1.0, done=True)

        return StepResult(state=new_state, reward=-1.0, done=False)

    @staticmethod
    def state_to_index(state: tuple[int, int]) -> int:
        r, c = state
        return r * N_COLS + c

    @staticmethod
    def n_states() -> int:
        return N_ROWS * N_COLS
