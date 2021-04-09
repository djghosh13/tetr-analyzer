# Tools for analysis of game state and actions

import numpy as np
from collections import defaultdict, namedtuple
from itertools import permutations
import sys

GRID_HEIGHT, GRID_WIDTH = 22, 10
VERBOSE = False

def _log(*args, **kwargs):
    if VERBOSE:
        print("[Solver] ", *args, **kwargs)

class SolverConfig:
    @staticmethod
    def set_verbose(verbose=True):
        global VERBOSE
        VERBOSE = verbose

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
    board = grid[::-1][garbage_height(grid):]
    erase = False
    for row in board:
        if erase:
            row[...] = "-"
        elif np.all(row == "-"):
            erase = True
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

def board_difference(ga, gb):
    ga, gb = to_array(ga), to_array(gb)
    diff = np.empty_like(ga, dtype=np.object)
    diff[...] = None
    neq = np.not_equal(ga, gb)
    diff[neq] = gb[neq]
    return diff

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

_piece_filters = {
    "S": to_filter([" **", "** "]),
    "Z": to_filter(["** ", " **"]),
    "T": to_filter([" * ", "***"]),
    "I": to_filter(["****"]),
    "L": to_filter(["  *", "***"]),
    "J": to_filter(["*  ", "***"]),
    "O": to_filter(["**", "**"])
}

# Placement

def apply_placement(board, piece, x, y, rotation):
        newb = to_array(board)
        ftr = _piece_filters[piece][rotation]
        Hf, Wf = ftr.shape
        newb[y:y + Hf, x:x + Wf][ftr] = piece
        # Check for line clears
        cleared = np.nonzero((newb != "-").all(axis=-1))[0]
        newb = to_array(
            ["-" * GRID_WIDTH] * len(cleared) +
            [row for i, row in enumerate(newb) if i not in cleared]
        )
        return newb

def all_placements(board, pieces):
    if not pieces:
        yield (), board
    else:
        piece, *rest = pieces
        filled = (board != "-")
        rotations = [4, 2, 2, 2, 1, 4, 4]["TSZIOLJ".index(piece)]
        for rot in range(rotations):
            ftr = _piece_filters[piece][rot]
            Hf, Wf = ftr.shape
            for y in range(GRID_HEIGHT - Hf, -1, -1):
                for x in range(GRID_WIDTH - Wf + 1):
                    # Check for validity and iterate
                    if filled[y:y + Hf, x:x + Wf][ftr].any(): continue
                    if y < GRID_HEIGHT - Hf and not filled[y + 1:y + Hf + 1, x:x + Wf][ftr].any(): continue
                    newboard = apply_placement(board, piece, x, y, rot)
                    for actions, result in all_placements(newboard, rest):
                        yield (Action(piece, x, y, rot),) + actions, result


def filter_pass(grid, filters):
    grid = to_array(grid)
    for i, f in enumerate(filters):
        Hf, Wf = f.shape
        X = np.pad(grid == "*", ((0, Hf - 1), (0, Wf - 1)), "constant")
        Xconv = np.lib.stride_tricks.as_strided(
            X,
            shape=(GRID_HEIGHT, GRID_WIDTH, Hf, Wf),
            strides=X.strides + X.strides
        )
        yield np.all(Xconv[:, :, f], axis=-1)


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
                frame["S"]["garbage"] = 0
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
        cpiece = piecelist[0]
        # First set (initial frame)
        first = next(frameiter)
        first["S"]["placed"], first["S"]["actions"] = (), ()
        yield first
        for start, frames in frameiter:
            target = frames[-1]
            # Compute candidate pieces
            cands = [cpiece, start["hold"]]
            cands.extend(piecelist[start["S"]["index"] + 1:target["S"]["index"] + 1])
            cands.remove(target["hold"])
            if "-" in cands:
                cands.remove("-")
            # Try all valid placements
            n_placed = sum(f["S"]["n_placed"] for f in frames)
            assert len(cands) == n_placed + 1 # All but new current piece were placed
            success = False
            for possibility in permutations(cands):
                *placed, leftover = possibility
                for actions, result in all_placements(start["S"]["board"], placed):
                    if board_equals(result, target["S"]["board"]):
                        success = True # SUCCESS!!!
                        for f in frames:
                            f["S"]["placed"] = tuple(placed[:f["S"]["n_placed"]])
                            f["S"]["actions"] = actions[:f["S"]["n_placed"]]
                            placed = placed[f["S"]["n_placed"]:]
                            actions = actions[f["S"]["n_placed"]:]
                        assert not placed
                        break
                if success:
                    cpiece = leftover
                    break
            if not success:
                _log("Could not work out piece placements")
                _log(f"Placed {n_placed} pieces out of {tuple(cands)}")
                _log("Start")
                pprint_board(start["S"]["board"])
                _log("Target")
                pprint_board(target["S"]["board"])
                raise Exception()
            yield from frames
        
    def reconstruct(self, frameiter, piecelist):
        for fn in (self.combine_frames, self.clean_frames, self.mark_good_frames,
                   self.piece_index, self.groupby_keyframe):
            frameiter = fn(frameiter)
        frameiter = self.pieces_placed(frameiter, piecelist)
        # Reconstruct events
        last = 0
        board = to_array(["-" * GRID_WIDTH] * GRID_HEIGHT)
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
                board = apply_placement(board, action.piece, action.x, action.y, action.rotation)
            if frame["S"]["garbage"]:
                yield {
                    "frame": frame["frame"],
                    "type": "garbage",
                    "event": Garbage(frame["S"]["garbage"]),
                    "board": board
                }
                board = apply_garbage(board, frame["S"]["garbage"])
            last = frame["end"]

    # Old stuff

    def calc_line_clears(self, frameiter):
        prev = None
        for frame in frameiter:
            frame["lines"] = []
            if prev is not None:
                nplaced = pieces_placed(prev, result)
                tilediff = board_change(prev["grid"], result["grid"])
                garbdiff = len(garbage_change(prev["grid"], result["grid"])[0])
                result["nplaced"] = nplaced
                result["garbage_cleared"] = garbdiff
                cleared = 4 * nplaced - garbdiff - tilediff
                if cleared % GRID_WIDTH:
                    _log(f"Warning: Uneven tile count ({prev['end']}-{result['frame']}): {cleared % GRID_WIDTH}", file=sys.stderr)
                    # print("\n".join("".join(row) for row in result["grid"]))
                cleared = int(np.ceil(cleared / GRID_WIDTH))
                # Figure out which lines were cleared
                curr, target = board_only(prev["grid"]), board_only(result["grid"])
                _blank = np.array(["-"] * GRID_WIDTH, dtype=np.object)
                for row in range(board_height(prev["grid"])):
                    currentrow = curr[row] if row < len(curr) else _blank
                    targetrow = target[row - len(result["lines"])] if row - len(result["lines"]) < len(target) else _blank
                    if np.all(currentrow[currentrow != "-"] == targetrow[currentrow != "-"]):
                        pass
                    else:
                        result["lines"].append(row)
                if len(result["lines"]) != cleared:
                    _log(f"Expected {cleared} lines, found {result['lines']}", file=sys.stderr)
            prev = result
            yield result

    def calc_piece_placements(self, frameiter):
        _blank = np.array(["-"] * GRID_WIDTH, dtype=np.object)
        prev = None
        for f in self.calc_line_clears(frameiter):
            result = f.copy()
            if prev is not None:
                grid = []
                # Get garbage
                for col in garbage_columns(result["grid"]):
                    grid.append(["-" if i == int(col) else "G" for i in range(GRID_WIDTH)])
                for col in garbage_columns(prev["grid"])[::-1][:result["garbage_cleared"]]:
                    grid.append(["*" if i == int(col) else "G" for i in range(GRID_WIDTH)])
                # Get lines
                gridA, gridB = board_only(prev["grid"]), board_only(result["grid"])
                j = 0
                for i in range(len(gridA)):
                    if i in result["lines"]:
                        row = gridA[i]
                        row[row == "-"] = "*"
                        grid.append(row)
                    else:
                        row = gridB[j] if j < len(gridB) else _blank
                        row[(gridA[i] == "-") & (row != "-")] = "*"
                        grid.append(row)
                        j += 1
                grid.extend(["-"] * GRID_WIDTH for _ in range(GRID_HEIGHT - len(grid)))
                grid = to_array(grid[:GRID_HEIGHT][::-1])
                result["grid"] = grid
                # Get placed piece(s)
                result["piece_placed"] = None
                # TODO: Add support for multiple piece placements?
                if result["nplaced"] == 1 and (grid == "*").sum() == 4:
                    for piece, filters in _piece_filters.items():
                        for rot, matches in enumerate(filter_pass(grid, filters)):
                            if np.any(matches):
                                result["piece_placed"] = {
                                    "piece": piece,
                                    "rotation": rot,
                                    "position": np.argwhere(matches)[0]
                                }
                                break
            prev = f
            yield result

    def calc_events(self, frameiter):
        prev = None
        for f in self.calc_piece_placements(frameiter):
            result = f.copy()
            # Lines cleared
            result["cleared"] = len(result["lines"]) + result.get("garbage_cleared", 0)
            # Combo
            result["combo"] = 0
            result["b2b"] = -1
            if prev is not None:
                if prev["cleared"]:
                    result["combo"] = prev["combo"] + 1
                # T spins
                result["tspin"] = False
                if result["piece_placed"] and result["piece_placed"]["piece"] == "T":
                    offset = ([1, 1], [1, 0], [0, 1], [1, 1])[result["piece_placed"]["rotation"]]
                    center = result["piece_placed"]["position"] + offset + 1
                    exgrid = ~np.pad(result["grid"] == "-", 1, "constant")
                    corners = exgrid[center[0]-1:center[0]+2:2, center[1]-1:center[1]+2:2]
                    front = (corners[:1], corners[:, 1:], corners[1:], corners[:, :1])[result["piece_placed"]["rotation"]]
                    if corners.sum() >= 3 and np.all(front):
                        result["tspin"] = True
                # Perfect clears
                result["perfect_clear"] = all((row == "-").sum() in (0, GRID_WIDTH) for row in result["grid"])
                # Back to back
                if result["cleared"]:
                    if result["tspin"] or result["cleared"] == 4:
                        result["b2b"] = prev["b2b"] + 1
                    else:
                        result["b2b"] = -1
                else:
                    result["b2b"] = prev["b2b"]
            # Calculate attack
            result["atk"] = defaultdict(int)
            if 1 <= result["cleared"] <= 4:
                result["atk"]["base"] = ([0, 2, 4, 6] if result["tspin"] else [0, 0, 1, 2, 4])[result["cleared"]]
                if result["atk"]["base"]:
                    result["atk"]["combo"] = result["combo"] * result["atk"]["base"] // 4
                else:
                    # Singles have different calculations
                    result["atk"]["combo"] = ([0] * 2 + [1] * 4 + [2] * 10 + [3] * 10)[result["combo"]]
                result["atk"]["b2b"] = (result["b2b"] >= 1) + (result["b2b"] >= 3) + (result["b2b"] >= 8)
                result["atk"]["pc"] = 10 * result["perfect_clear"]
            result["attack"] = sum(result["atk"].values())
            result["attack_combo"] = 0
            if prev is not None and result["attack"]:
                result["attack_combo"] = result["attack"] + prev["attack_combo"]
            prev = result
            yield result

    def find_interesting_parts(self, framelist):
        # TODO: Work in progress
        goodframes = set()
        for i in range(7, len(framelist)):
            if framelist[i]["attack_combo"] >= 10 and framelist[i]["combo"] > 1:
                for ii in range(i - 5, i + 1):
                    goodframes.add(ii)
        for i in sorted(goodframes):
            self.log.append(framelist[i]["frame"])



class SolverStats:
    @staticmethod
    def compute_placement_stats(framelist, out):
        for f in framelist:
            if "piece_placed" in f and f["piece_placed"] is not None:
                out.setdefault(f["piece_placed"]["piece"], {}).setdefault(f["piece_placed"]["rotation"], 0)
                out[f["piece_placed"]["piece"]][f["piece_placed"]["rotation"]] += 1

    @staticmethod
    def visualize_placements(stats):
        nrots = { "S": 2, "Z": 2, "T": 4, "L": 4, "J": 4, "I": 2, "O": 1 }
        res = []
        for piece in nrots:
            placements = stats.get(piece, {})
            for rot in range(nrots[piece]):
                pstrs = ["".join([" ", piece][x] for x in row) for row in _piece_filters[piece][rot]]
                pstrs[0] = f"{pstrs[0]: <4}  {placements.get(rot, 0): >3}"
                res.append("\n".join(pstrs))
        return "\n\n".join(res)

    @staticmethod
    def tspin_overhangs(framelist, out):
        for f in framelist:
            # Only point down T spins
            if f["cleared"] > 0 and f["tspin"] and f["piece_placed"]["rotation"] == 2:
                ca, cb = f["piece_placed"]["position"] + [-1, 0], f["piece_placed"]["position"] + [-1, 2]
                out[f["grid"][ca[0]][ca[1]]] += 1
                out[f["grid"][cb[0]][cb[1]]] += 1

    @staticmethod
    def tspin_columns(framelist, out):
        for f in framelist:
            # Only point down T spins
            if f["cleared"] > 0 and f["tspin"] and f["piece_placed"]["rotation"] == 2:
                col = f["piece_placed"]["position"][1] + 1
                out[col] += 1
                