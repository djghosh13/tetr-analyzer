from typing import AnyStr, Generator
import functools
import numpy as np
import time
from tqdm import tqdm

import solver_tools as st
from solver_tools import GRID_HEIGHT, GRID_WIDTH, apply_placement
from render_tools import RendererConfig, ReplayVideo
from ai_routines import BasicAI, FourWideAI, QuadAI, TetrisAI


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

def nextpiece_generator(seed: int = None, allow_szo_start=True):
    rng: np.random.Generator = np.random.default_rng(seed)
    bag = rng.permutation(BAG)
    while not allow_szo_start and bag[0] in "SZO":
        bag = rng.permutation(BAG)
    yield from bag
    while True:
        bag = rng.permutation(BAG)
        yield from bag


def play_game(agent: TetrisAI, board: np.ndarray, generator: Generator[AnyStr, None, None]):
    queue = []
    agent.reset()
    for piece in tqdm(generator):
        queue.append(piece)
        action = agent.next_action(board, queue)
        if action is not None:
            newboard = apply_placement(board, action.piece, action.x, action.y, action.rotation)
            lines = st.lines_cleared(board, newboard, 1)
            board = newboard
            yield action
            queue.pop(0 if action.piece == queue[0] else 1)
            agent.step(board, lines)


def nline_race(agent, n, seed=None, fpp=60):
    generator = nextpiece_generator(seed)
    board = np.empty([GRID_HEIGHT, GRID_WIDTH], dtype=np.object)
    board[...] = "-"
    lines = 0
    video = ReplayVideo()
    for idx, action in enumerate(play_game(agent, board, generator), 1):
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


def cheese_race(agent, n, seed=None, fpp=60):
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
    for idx, action in enumerate(play_game(agent, board, generator), 1):
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


def marathon(agent, n, seed=None, fpp=60):
    generator = nextpiece_generator(seed)
    board = np.empty([GRID_HEIGHT, GRID_WIDTH], dtype=np.object)
    board[...] = "-"
    video = ReplayVideo()
    for idx, action in enumerate(play_game(agent, board, generator), 1):
        video.render({
            "frame": idx * fpp,
            "type": "drop",
            "event": action,
            "board": board
        })
        try:
            board = st.apply_placement(board, action.piece, action.x, action.y, action.rotation)
        except IndexError:
            break
        if idx >= n:
            break
    video.extend_by(2 * fpp)
    return video


if __name__ == "__main__":
    RendererConfig.set_scale(20)

    agent = FourWideAI()
    timer(nline_race)(agent, 40, seed=42, fpp=12).save("videos/mygame.mp4")
    # timer(cheese_race)(agent, 18, seed=42, fpp=12).save("videos/mygame.mp4")
    # timer(marathon)(agent, 140, seed=42, fpp=12).save("videos/mygame.mp4")