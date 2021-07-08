import functools
from render_tools import RendererConfig, ReplayVideo
import numpy as np
import time

import solver_tools as st
from solver_tools import GRID_HEIGHT, GRID_WIDTH


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return value
    return wrapper_timer


BAG = list(st.piece_filters.keys())

def nextpiece_generator(seed: int = None):
    rng: np.random.Generator = np.random.default_rng(seed)
    while True:
        bag = rng.permutation(BAG)
        yield from bag


def harddrop_placements(board, pieces):
    if not pieces:
        yield (), board
    else:
        piece, *rest = pieces
        filled = (board != "-")
        heights = (GRID_HEIGHT - np.argmax(filled, axis=0)) * np.any(filled, axis=0)
        rotations = [4, 2, 2, 2, 1, 4, 4]["TSZIOLJ".index(piece)]
        for rot in range(rotations):
            ftr = st.piece_filters[piece][rot]
            Hf, Wf = ftr.shape
            bases = np.argmax(ftr[::-1], axis=0)
            for x in range(GRID_WIDTH - Wf + 1):
                y = GRID_HEIGHT - Hf - np.max(heights[x:x + Wf] - bases)
                # Check for validity and iterate
                if y < 0: continue
                newboard = st.apply_placement(board, piece, x, y, rot)
                for actions, result in harddrop_placements(newboard, rest):
                    yield (st.Action(piece, x, y, rot),) + actions, result
                    

def lines_cleared(boardA: np.ndarray, boardB: np.ndarray, npieces: int):
    return (np.sum(boardA != "-") + 4 * npieces - np.sum(boardB != "-")) // st.GRID_WIDTH

def fitness_heuristic(board, lines):
    # From https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/
    HEIGHT_FACTOR, HOLES_FACTOR, BUMPINESS_FACTOR, LINES_FACTOR = -0.510, -0.357, -0.184, 0.761
    # HEIGHT_FACTOR, HOLES_FACTOR, BUMPINESS_FACTOR, LINES_FACTOR = -2.510, -10.0, -0.184, 10.0
    # Aggregate height (above threshold)
    heights = GRID_HEIGHT - np.argmax(board != "-", axis=0)
    heights *= np.any(board != "-", axis=0)
    aggheight = np.clip(heights - 0, 0, None).sum()
    # Number of holes
    nholes = heights.sum() - np.sum(board != "-")
    # Bumpiness
    bumpiness = np.abs(heights[1:] - heights[:-1])
    bumpiness = bumpiness.sum() - np.sum(sorted(bumpiness)[-1:])
    # Weighted combination of factors
    return HEIGHT_FACTOR * aggheight + HOLES_FACTOR * nholes + BUMPINESS_FACTOR * bumpiness + LINES_FACTOR * lines


def play_game(board, generator):
    LOOKAHEAD = 2
    queue = []
    for piece in generator:
        queue.append(piece)
        if len(queue) < LOOKAHEAD: continue
        bestactions, bestboard, bestfitness = None, None, -np.inf
        hasactions = False
        for actions, result in harddrop_placements(board, queue):
            hasactions = True
            fitness = fitness_heuristic(result, lines_cleared(board, result, LOOKAHEAD))
            if fitness > bestfitness:
                bestactions, bestboard, bestfitness = actions, result, fitness
        if not hasactions:
            print("No actions found")
        board = bestboard
        yield from bestactions
        queue.clear()


def nline_race(n, seed=None, fpp=60):
    generator = nextpiece_generator(seed)
    board = np.empty([GRID_HEIGHT, GRID_WIDTH], dtype=np.object)
    board[...] = "-"
    lines = 0
    video = ReplayVideo()
    for idx, action in enumerate(play_game(board, generator), 1):
        video.render({
            "frame": idx * fpp,
            "type": "drop",
            "event": action,
            "board": board
        })
        tiles = (board != "-").sum() + 4
        board = st.apply_placement(board, action.piece, action.x, action.y, action.rotation)
        lines += (tiles - (board != "-").sum()) // 10
        if lines >= n:
            break
    video.extend_by(2 * fpp)
    return video


def cheese_race(n, seed=None, fpp=60):
    generator = nextpiece_generator(seed)
    board = np.empty([GRID_HEIGHT, GRID_WIDTH], dtype=np.object)
    board[...] = "-"
    # Create cheesy board
    board[-n:] = "G"
    rng = np.random.default_rng(seed)
    column = rng.integers(GRID_WIDTH)
    for i in range(n):
        board[-i - 1, column] = "-"
        column = (column + rng.integers(1, GRID_WIDTH)) % GRID_WIDTH
    # Play
    video = ReplayVideo()
    for idx, action in enumerate(play_game(board, generator), 1):
        video.render({
            "frame": idx * fpp,
            "type": "drop",
            "event": action,
            "board": board
        })
        board = st.apply_placement(board, action.piece, action.x, action.y, action.rotation)
        if not np.any(board == "G"):
            print(f"Used {idx} pieces")
            break
    video.extend_by(2 * fpp)
    return video


if __name__ == "__main__":
    RendererConfig.set_scale(20)
    timer(nline_race)(40, fpp=12).save("videos/mygame.mp4")
    # timer(cheese_race)(10, fpp=12).save("videos/mygame.mp4")