# Tools for analysis of game state and actions

import numpy as np
from collections import defaultdict, namedtuple
from itertools import combinations
import sys


GRID_HEIGHT, GRID_WIDTH = 22, 10
VERBOSE = False
IGNORE_ERRORS = False

def _log(*args, **kwargs):
    if VERBOSE:
        print("[Solver] ", *args, **kwargs)

class SolverConfig:
    @staticmethod
    def set_verbose(verbose=True):
        global VERBOSE
        VERBOSE = verbose
        return SolverConfig

    @staticmethod
    def set_force(force=True):
        global IGNORE_ERRORS
        IGNORE_ERRORS = force
        return SolverConfig

def to_array(grid):
    return grid.copy() if isinstance(grid, np.ndarray) else np.array([list(row) for row in grid], dtype=np.object)

# Garbage analysis

def garbage_height(grid):
    grid = to_array(grid)
    for i in range(len(grid)):
        if (grid[-i - 1] == "G").sum() != GRID_WIDTH - 1:
            return i
    return len(grid)

def garbage_columns(grid):
    grid = to_array(grid)
    columns = ""
    for row in grid[::-1]:
        if (row == "G").sum() == GRID_WIDTH - 1:
            columns += str(np.argmax(row != "G"))
        else:
            break
    return columns

def garbage_change(gridA, gridB):
    gridA, gridB = to_array(gridA), to_array(gridB)
    gA, gB = garbage_columns(gridA), garbage_columns(gridB)
    cleared = ""
    while not gB.endswith(gA):
        cleared, gA = cleared + gA[-1], gA[:-1]
    added = gB[:-len(gA)] if gA else gB
    return cleared, added

def erase_garbage(grid):
    grid = to_array(grid)
    board = grid[::-1][garbage_height(grid):][::-1]
    newgrid = np.empty_like(grid)
    newgrid[...] = "-"
    newgrid[::-1][:len(board)][::-1] = board
    return newgrid

def apply_garbage(board, columns):
    garbage = [["-" if i == int(col) else "G" for i in range(GRID_WIDTH)] for col in columns]
    return to_array(list(board[len(columns):]) + garbage[::-1])

# Board analysis

def board_clean(grid):
    grid = to_array(grid)
    # Delete pieces in garbage
    gbrows = grid[::-1][:garbage_height(grid)]
    gbrows[gbrows != "G"] = "-"
    # Delete floating rows
    board = grid[::-1][garbage_height(grid):]
    erase = False
    for row in board:
        if erase:
            row[...] = "-"
        elif np.all(row == "-"):
            erase = True
    # Special case, might help?
    grid[:1, 3:7] = "-"
    return grid

def board_tiles(grid):
    return np.sum((grid != "-") & (grid != "G"))

def board_only(grid):
    grid = board_clean(grid)
    return grid[::-1][garbage_height(grid):]

def board_intersection(ga, gb):
    ga, gb = to_array(ga), to_array(gb)
    if garbage_columns(ga) != garbage_columns(gb):
        _log("Warning: Mismatched garbage columns", file=sys.stderr)
    ga[gb == "-"] = "-"
    return ga

def board_change(gridA, gridB):
    gridA, gridB = to_array(gridA), to_array(gridB)
    boardA, boardB = board_only(gridA), board_only(gridB)
    tilesA = {v:count for v, count in zip(*np.unique(boardA, return_counts=True)) if v != "-"}
    tilesB = {v:count for v, count in zip(*np.unique(boardB, return_counts=True)) if v != "-"}
    tilediff = sum(tilesB.values()) - sum(tilesA.values())
    return tilediff

def board_height(grid):
    board = board_only(grid)
    for i, row in enumerate(board):
        if np.all(row == "-"):
            return i
    return len(board)

def board_equals(gridA, gridB):
    garbdiff = np.sum(gridB == "G") - np.sum(gridA == "G")
    if garbdiff < 0 or garbdiff % (GRID_WIDTH - 1):
        return False
    garbdiff //= GRID_WIDTH - 1
    boardA, boardB = gridA[garbdiff:], gridB[:len(gridB) - garbdiff]
    return np.all(boardA == boardB)

def board_difference(gridA, gridB):
    garbdiff = np.sum(gridB == "G") - np.sum(gridA == "G")
    if garbdiff < 0 or garbdiff % (GRID_WIDTH - 1):
        return np.inf
    garbdiff //= GRID_WIDTH - 1
    boardA, boardB = gridA[garbdiff:], gridB[:len(gridB) - garbdiff]
    return np.sum(boardA != boardB)

# Piece analysis

Action = namedtuple("Action", ["piece", "x", "y", "rotation"])
Garbage = namedtuple("Garbage", ["columns"])

def action_taken(frameA, frameB):
    return frameA["hold"] != frameB["hold"] or frameA["next"] != frameB["next"]

def pieces_placed(frameA, frameB):
    na, ha = frameA["next"], frameA["hold"]
    nb, hb = frameB["next"], frameB["hold"]
    while not nb.startswith(na):
        na = na[1:]
    if not na:
        raise RuntimeError("Too many frames were dropped")
    return 5 - len(na) - (ha == "-" and hb != "-")

def to_filter(fshape):
    rotate = lambda x: x[::-1].T
    fltr = to_array(fshape) == "*"
    return [
        fltr,
        rotate(fltr),
        rotate(rotate(fltr)),
        rotate(rotate(rotate(fltr)))
    ]

piece_filters = {
    "S": to_filter([" **", "** "]),
    "Z": to_filter(["** ", " **"]),
    "T": to_filter([" * ", "***"]),
    "I": to_filter(["****"]),
    "L": to_filter(["  *", "***"]),
    "J": to_filter(["*  ", "***"]),
    "O": to_filter(["**", "**"])
}

# Placement

def all_orderings(n, queue, h_initial, h_final):
    for nholds in range(n + 2):
        for ishold in combinations(range(n + 1), nholds):
            q, h = tuple(queue), h_initial
            ordering = []
            for idx in range(n + 1):
                if idx in ishold:
                    if h == "-":
                        h, q = q[0], q[1:]
                    else:
                        h, q = q[0], (h,) + q[1:]
                ordering.append(q[0])
                q = q[1:]
            if h == h_final:
                yield ordering

def apply_placement(board, piece, x, y, rotation):
    newb = to_array(board)
    ftr = piece_filters[piece][rotation]
    Hf, Wf = ftr.shape
    assert np.all(newb[y:y + Hf, x:x + Wf][ftr] == "-")
    newb[y:y + Hf, x:x + Wf][ftr] = piece
    # Check for line clears
    cleared = np.nonzero((newb != "-").all(axis=-1))[0]
    newb = to_array(
        ["-" * GRID_WIDTH] * len(cleared) +
        [row for i, row in enumerate(newb) if i not in cleared]
    )
    return newb

def all_placements(board, pieces, states):
    if not pieces:
        yield (), board
    else:
        piece, *rest = pieces
        filled = (board != "-")
        rotations = [4, 2, 2, 2, 1, 4, 4]["TSZIOLJ".index(piece)]
        for rot in range(rotations):
            ftr = piece_filters[piece][rot]
            Hf, Wf = ftr.shape
            for y in range(GRID_HEIGHT - Hf, -1, -1):
                for x in range(GRID_WIDTH - Wf + 1):
                    # Check for validity and iterate
                    if filled[y:y + Hf, x:x + Wf][ftr].any(): continue
                    if y < GRID_HEIGHT - Hf and not filled[y + 1:y + Hf + 1, x:x + Wf][ftr].any(): continue
                    newboard = apply_placement(board, piece, x, y, rot)
                    # Compare to heuristic if exists
                    if states and states[0] is not None:
                        if board_difference(newboard, states[0]) > 4:
                            continue
                    for actions, result in all_placements(newboard, rest, states=states[1:]):
                        yield (Action(piece, x, y, rot),) + actions, result


def pprint_board(board):
    print("\n".join("".join(row) for row in board))
    print()

# TODO: Solution

class Solver:
    def __init__(self):
        self.events = []
        self.log = []

    def groupby_action(self, frameiter):
        curr = []
        for f in frameiter:
            if not curr:
                curr.append(f)
            elif action_taken(curr[-1], f):
                yield curr
                curr = [f]
            else:
                curr.append(f)
        if curr:
            yield curr

    def combine_frames(self, frameiter):
        for frames in self.groupby_action(frameiter):
            result = frames[0].copy()
            result["S"] = {
                "board": to_array(result["grid"])
            }
            for f in frames:
                result["S"]["board"] = board_intersection(result["S"]["board"], f["grid"])
                result["end"] = f["frame"]
            yield result

    def clean_frames(self, frameiter):
        for frame in frameiter:
            frame["S"]["board"] = board_clean(frame["S"]["board"])
            yield frame

    def mark_good_frames(self, frameiter):
        prev, balance = {}, 0
        for frame in frameiter:
            if prev:
                frame["S"]["n_placed"] = pieces_placed(prev, frame)
                gclear, gadd = garbage_change(prev["S"]["board"], frame["S"]["board"])
                frame["S"]["garbage"] = gadd
                # Change in tiles offset by placements and garbage lines
                balance += 4 * frame["S"]["n_placed"]
                balance -= board_tiles(frame["S"]["board"]) - board_tiles(prev["S"]["board"])
                balance -= len(gclear)
                balance %= GRID_WIDTH
            else:
                frame["S"]["n_placed"] = 0
                frame["S"]["garbage"] = ""
            frame["S"]["balance"] = balance
            frame["S"]["keyframe"] = (balance == 0)
            prev = frame
            yield frame

    def compute_piece_list(self, framelist):
        pieces = ""
        for frame in framelist:
            nextps = frame["next"]
            for k in range(5, -1, -1):
                if pieces.endswith(nextps[:k]):
                    pieces = pieces + nextps[k:]
                    break
        firstpiece = list(set("TSZIOLJ") - set(pieces[:6]))[0]
        return firstpiece + pieces

    def piece_index(self, frameiter):
        prev, idx = {}, 0
        for frame in frameiter:
            idx += frame["S"]["n_placed"]
            if prev and prev["hold"] == "-" and frame["hold"] != "-":
                idx += 1
            frame["S"]["index"] = idx
            prev = frame
            yield frame

    def groupby_keyframe(self, frameiter):
        curr, start = [], None
        # Initial frame
        start = next(frameiter)
        yield start
        # Rest
        for f in frameiter:
            curr.append(f)
            if f["S"]["keyframe"]:
                yield start, curr
                curr, start = [], f

    def pieces_placed(self, frameiter, piecelist):
        # First set (initial frame)
        first = next(frameiter)
        first["S"]["placed"], first["S"]["actions"] = (), ()
        yield first
        #
        cpiece = piecelist[0]
        prevstart, prevframes = None, None
        for start, frames in frameiter:
            if prevstart is not None:
                start, frames = prevstart, prevframes + frames
            target = frames[-1]
            # Compute candidate pieces
            cands = [cpiece]
            cands.extend(piecelist[start["S"]["index"] + 1:target["S"]["index"] + 1])
            # Try all valid placements
            n_placed = sum(f["S"]["n_placed"] for f in frames)
            success = False
            # Perform search with heuristic pruning if too many pieces
            if n_placed > 3:
                _log("More than 3 pieces dropped, using intermediate states")
                heuristic = []
                for f in frames:
                    if f["S"]["n_placed"]:
                        heuristic += [None] * (f["S"]["n_placed"] - 1)
                        heuristic.append(f["S"]["board"])
            else:
                heuristic = [None] * n_placed
            # Check all possibilities
            for possibility in all_orderings(n_placed, cands, start["hold"], target["hold"]):
                *placed, leftover = possibility
                for actions, result in all_placements(start["S"]["board"], placed, states=heuristic):
                    if board_equals(result, target["S"]["board"]):
                        success = True
                        for f in frames:
                            f["S"]["placed"] = tuple(placed[:f["S"]["n_placed"]])
                            f["S"]["actions"] = actions[:f["S"]["n_placed"]]
                            placed = placed[f["S"]["n_placed"]:]
                            actions = actions[f["S"]["n_placed"]:]
                        assert not placed
                        break
                if success:
                    cpiece = leftover
                    if prevstart is not None:
                        _log("Resolved dropped keyframe")
                        prevstart, prevframes = None, None
                    break
            if not success:
                if n_placed > 3:
                    raise TimeoutError("Too many pieces to try all solutions")
                _log(f"Warning: Failed to solve interval {start['frame']} to {target['frame']}")
                prevstart, prevframes = start, frames
            if success:
                yield from frames
        
    def reconstruct(self, frameiter, piecelist):
        for fn in (self.combine_frames, self.clean_frames, self.mark_good_frames,
                   self.piece_index, self.groupby_keyframe):
            frameiter = fn(frameiter)
        frameiter = self.pieces_placed(frameiter, piecelist)
        # Reconstruct events
        last = 0
        board = to_array(["-" * GRID_WIDTH] * GRID_HEIGHT)
        try:
            for frame in frameiter:
                n = frame["S"]["n_placed"]
                step = (frame["frame"] - last) / (n + 1)
                for i, action in enumerate(frame["S"]["actions"]):
                    yield {
                        "frame": int(np.ceil(last + step * (i + 1))),
                        "type": "drop",
                        "event": action,
                        "board": board
                    }
                    try:
                        board = apply_placement(board, action.piece, action.x, action.y, action.rotation)
                    except AssertionError as e:
                        print(f"Frame {last}: {action}")
                        pprint_board(board)
                        raise e
                if frame["S"]["keyframe"]:
                    board = to_array(frame["S"]["board"])
                # if frame["S"]["garbage"]:
                #     yield {
                #         "frame": frame["frame"],
                #         "type": "garbage",
                #         "event": Garbage(frame["S"]["garbage"]),
                #         "board": board
                #     }
                #     board = apply_garbage(board, frame["S"]["garbage"])
                last = frame["end"]
        except (AssertionError, RuntimeError, TimeoutError) as e:
            if IGNORE_ERRORS:
                _log(e)
                return
            else:
                raise e

    # Old stuff

    def compute_attacks(self, events):
        combo, b2b, spike = 0, 0, 0
        for ev in events:
            if ev["type"] == "garbage": continue
            action = ev["event"]
            boardA = ev["board"]
            boardB = apply_placement(boardA, action.piece, action.x, action.y, action.rotation)
            # Compute base statistics
            ev["A"] = {}
            lines = (boardA != "-").sum() + 4 - (boardB != "-").sum()
            assert lines % GRID_WIDTH == 0
            lines //= GRID_WIDTH
            ev["A"]["lines"] = lines
            ev["A"]["combo"], ev["A"]["b2b"] = 0, 0
            ev["A"]["tetris"], ev["A"]["tspin"], ev["A"]["mini_tspin"] = False, False, False
            ev["A"]["perfect_clear"] = lines and np.all(boardB == "-")
            ev["A"]["garbage"] = len(garbage_change(boardA, boardB)[0])
            if lines:
                combo += 1
                ev["A"]["combo"] = combo - 1
                ev["A"]["tetris"] = (lines == 4)
                if action.piece == "T":
                    cx = action.x + (action.rotation != 1) + 1
                    cy = action.y + (action.rotation != 2) + 1
                    exgrid = ~np.pad(boardA == "-", 1, "constant")
                    corners = exgrid[cy - 1:cy + 2:2, cx - 1:cx + 2:2]
                    front = (corners[:1], corners[:, 1:], corners[1:], corners[:, :1])[action.rotation]
                    if corners.sum() >= 3:
                        if np.all(front):
                            ev["A"]["tspin"] = True
                        else:
                            ev["A"]["mini_tspin"] = True
                if ev["A"]["tetris"] or ev["A"]["tspin"]: # or ev["A"]["mini_tspin"]:
                    b2b += 1
                    ev["A"]["b2b"] = b2b - 1
                else:
                    b2b = 0
            else:
                combo = 0
            # Calculate attack
            ev["A"]["attack"] = defaultdict(int)
            if ev["A"]["lines"]:
                ev["A"]["attack"]["base"] = ([0, 2, 4, 6] if ev["A"]["tspin"] else [0, 0, 1, 2, 4])[ev["A"]["lines"]]
                if ev["A"]["attack"]["base"]:
                    ev["A"]["attack"]["combo"] = ev["A"]["combo"] * ev["A"]["attack"]["base"] // 4
                else:
                    # Singles have different calculations
                    ev["A"]["attack"]["combo"] = ([0] * 2 + [1] * 4 + [2] * 10 + [3] * 10)[ev["A"]["combo"]]
                ev["A"]["attack"]["b2b"] = (ev["A"]["b2b"] >= 1) + (ev["A"]["b2b"] >= 3) + (ev["A"]["b2b"] >= 8)
                ev["A"]["attack"]["pc"] = 10 * ev["A"]["perfect_clear"]
            ev["A"]["total_attack"] = sum(ev["A"]["attack"].values())
            #
            if ev["A"]["total_attack"]:
                spike += ev["A"]["total_attack"]
            else:
                spike = 0
            ev["A"]["spike"] = spike
            yield ev


class SolverStats:
    @staticmethod
    def compute_placement_stats(events, out):
        for ev in events:
            if ev["type"] == "garbage": continue
            action = ev["event"]
            out.setdefault(action.piece, defaultdict(int))[action.rotation] += 1

    @staticmethod
    def visualize_placements(stats):
        nrots = { "S": 2, "Z": 2, "T": 4, "L": 4, "J": 4, "I": 2, "O": 1 }
        res = []
        for piece in nrots:
            placements = stats.get(piece, {})
            for rot in range(nrots[piece]):
                pstrs = ["".join([" ", piece][x] for x in row) for row in piece_filters[piece][rot]]
                pstrs[0] = f"{pstrs[0]: <4}  {placements[rot]: >3}"
                res.append("\n".join(pstrs))
        return "\n\n".join(res)

    @staticmethod
    def tspin_overhangs(events, out):
        for ev in events:
            if ev["type"] == "garbage": continue
            # Only point down T spins
            if ev["A"]["lines"] and ev["A"]["tspin"] and ev["event"].rotation == 2:
                lx, rx, y = ev["event"].x, ev["event"].x + 2, ev["event"].y - 1
                out[ev["board"][y][lx]] += 1
                out[ev["board"][y][rx]] += 1

    @staticmethod
    def tspin_columns(events, out):
        for ev in events:
            if ev["type"] == "garbage": continue
            # Only point down T spins
            if ev["A"]["lines"] and ev["A"]["tspin"] and ev["event"].rotation == 2:
                col = ev["event"].x + 1
                out[col] += 1
