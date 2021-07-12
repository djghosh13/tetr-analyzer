from typing import Any, List
import numpy as np
from collections import namedtuple
import heapq
from dataclasses import dataclass, field
from itertools import chain

from solver_tools import GRID_HEIGHT, GRID_WIDTH, all_placements, harddrop_placements, lines_cleared


SearchState = namedtuple("SearchState", ["board", "pieces", "action", "netreward", "prev"])

@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: SearchState = field(compare=False)

class TetrisAI:
    LOOKAHEAD = 6
    BEAM_WIDTH = 8
    HARD_DROP = True

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key.upper()):
                setattr(self, key.upper(), value)

    def reset(self):
        """Tell the AI the game is (re)starting."""
        pass

    def step(self, board: np.ndarray, lines: int):
        """Tell the AI a piece was placed."""
        pass

    def fitness(self, board: np.ndarray) -> float:
        """Fitness heuristic. Can be calculated solely using the board state."""
        pass

    def reward(self, board: np.ndarray, previous: np.ndarray) -> float:
        """Reward function. Need to know the change in board state to compute (e.g. lines cleared)."""
        pass

    def next_action(self, board: np.ndarray, pieces: List[Any]):
        if len(pieces) < self.LOOKAHEAD + 1:
            return None
        return self._beam_search(board, pieces)

    def _beam_search(self, board: np.ndarray, pieces: List[Any]):
        placement_fn = harddrop_placements if self.HARD_DROP else all_placements
        # Perform beam search
        states = [PrioritizedItem(0, SearchState(board, pieces, None, 0, None))]
        for _ in range(self.LOOKAHEAD):
            nextstates = []
            for pqentry in states:
                state = pqentry.item
                pieceH, pieceN, *rest = state.pieces
                for (action,), result in chain(placement_fn(state.board, [pieceN], [None]), placement_fn(state.board, [pieceH], [None])):
                    netreward = self.reward(result, state.board) + state.netreward
                    priority = self.fitness(result) + netreward
                    nextpieces = [pieceH if action.piece == pieceN else pieceN] + rest
                    heapq.heappush(nextstates, PrioritizedItem(priority, SearchState(result, nextpieces, action, netreward, state)))
                    if len(nextstates) > self.BEAM_WIDTH:
                        heapq.heappop(nextstates)
            states = nextstates
        # Select best
        beststate = heapq.nlargest(1, states)[0].item
        while beststate.prev.prev is not None:
            beststate = beststate.prev
        return beststate.action


class BasicAI(TetrisAI):
    # From https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/
    # HEIGHT_FACTOR, HOLES_FACTOR, BUMPINESS_FACTOR, LINES_FACTOR = -0.510, -0.357, -0.184, 0.761
    HEIGHT_FACTOR, HOLES_FACTOR, BUMPINESS_FACTOR, LINES_FACTOR = -0.510, -100.0, -0.184, 1.0
    HEIGHT_THRESHOLD = 4

    def fitness(self, board):
        # Aggregate height (above threshold)
        heights = column_heights(board)
        aggheight = np.clip(heights - self.HEIGHT_THRESHOLD, 0, None).sum()
        # Number of holes
        nholes = heights.sum() - np.sum(board != "-")
        # Bumpiness
        bumpiness = np.abs(heights[1:] - heights[:-1])
        bumpiness = bumpiness.sum() - np.sum(sorted(bumpiness)[-1:])
        # Weighted combination of factors
        return (
            self.HEIGHT_FACTOR * aggheight +
            self.HOLES_FACTOR * nholes +
            self.BUMPINESS_FACTOR * bumpiness
        )
    
    def reward(self, board, previous):
        # Lines cleared
        lines = lines_cleared(previous, board, 1)
        return self.LINES_FACTOR * lines


class QuadAI(TetrisAI):
    HEIGHT_FACTOR, HOLES_FACTOR, BUMPINESS_FACTOR, LINES_FACTOR = -0.510, -100.0, -0.184, 1.0
    HEIGHT_THRESHOLD = 4
    SKIM_PENALTY = 100

    def fitness(self, board):
        # Aggregate height (above threshold)
        heights = column_heights(board)
        aggheight = np.clip(heights - self.HEIGHT_THRESHOLD, 0, None).sum()
        # Number of holes
        nholes = heights.sum() - np.sum(board != "-")
        # Bumpiness
        bumpiness = np.abs(heights[1:] - heights[:-1])
        bumpiness = bumpiness.sum() - np.sum(sorted(bumpiness)[-1:])
        # Weighted combination of factors
        return (
            self.HEIGHT_FACTOR * aggheight +
            self.HOLES_FACTOR * nholes +
            self.BUMPINESS_FACTOR * bumpiness
        )
    
    def reward(self, board, previous):
        # Lines cleared (but incentivize quads only)
        lines = lines_cleared(previous, board, 1)
        return self.LINES_FACTOR * lines - self.SKIM_PENALTY * (lines not in [0, 4])


class FourWideAI(TetrisAI):
    HEIGHT_FACTOR, HOLES_FACTOR, BUMPINESS_FACTOR = -0.510, -100.0, -0.184
    HEIGHT_THRESHOLD = 8
    FOUR_WIDE_PENALTY = -1000
    SKIM_PENALTY, SKIM_REWARD = -10, 40

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.comboing = False
        self.comboheight = 0

    def reset(self):
        self.comboing = False
        self.HARD_DROP = not self.comboing

    def step(self, board, lines):
        columnheight = np.all(board[:, :-4] != "-", axis=1).sum()
        if self.comboing and columnheight < 4:
            self.comboing = False
        elif not self.comboing and columnheight > 18:
            self.comboing = True
        self.HARD_DROP = not self.comboing

    def fitness(self, board):
        # Aggregate height (above threshold)
        heights = column_heights(board)
        aggheight = np.clip(heights - self.HEIGHT_THRESHOLD, 0, None).sum()
        # Number of holes
        nholes = heights.sum() - np.sum(board != "-")
        # Bumpiness
        bumpiness = np.abs(heights[1:] - heights[:-1]) ** 2
        bumpiness = bumpiness.sum() - bumpiness[-4]
        # Allow only 3 tiles in 4 right columns
        penalty = ((board[:, -4:] != "-").sum() != 3)
        # Weighted combination of factors
        return (
            self.HEIGHT_FACTOR * aggheight +
            self.HOLES_FACTOR * nholes +
            self.BUMPINESS_FACTOR * bumpiness +
            self.FOUR_WIDE_PENALTY * penalty
        )
    
    def reward(self, board, previous):
        # Lines cleared (decentivize)
        lines = lines_cleared(previous, board, 1)
        if self.comboing:
            return self.SKIM_REWARD * (lines > 0)
        else:
            return self.SKIM_PENALTY * lines

# Utility functions

def column_heights(board: np.ndarray) -> np.ndarray:
    filled = (board != "-")
    return (GRID_HEIGHT - np.argmax(filled, axis=0)) * np.any(filled, axis=0)