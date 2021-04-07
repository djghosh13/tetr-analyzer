# Tools for analysis of game state and actions

import numpy as np
from collections import defaultdict
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

# Piece analysis

def pieces_placed(frameA, frameB):
    na, ha = frameA["next"], frameA["hold"]
    nb, hb = frameB["next"], frameB["hold"]
    while not nb.startswith(na):
        na = na[1:]
    if not na:
        raise RuntimeError("Too many frames were dropped")
    return 5 - len(na) - (ha == "-" and hb != "-")

def which_pieces(frameA, frameB):
    try:
        na, ha = frameA["next"], frameA["hold"]
        nb, hb = frameB["next"], frameB["hold"]
        pieces = []
        if ha != "-":
            pieces.append(ha)
        while not nb.startswith(na):
            pieces.append(na[0])
            na = na[1:]
        if hb != "-":
            pieces.remove(hb)
    except Exception:
        print(frameA["next"], frameA["hold"])
        print(frameB["next"], frameB["hold"])
        sys.exit(1)
    return pieces

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

# TODO: Solution

class Solver:
    def __init__(self):
        self.events = []
        self.log = []

    def groupby_placement(self, frameiter):
        curr = []
        for f in frameiter:
            if not curr:
                curr.append(f)
            elif pieces_placed(curr[-1], f):
                yield curr
                curr = [f]
            else:
                curr.append(f)
        if curr:
            yield curr

    def combine_frames(self, frameiter):
        for frames in self.groupby_placement(frameiter):
            result = frames[0].copy()
            for f in frames:
                result["grid"] = board_intersection(result["grid"], f["grid"])
                result["end"] = f["frame"]
            yield result

    def clean_frames(self, frameiter):
        for f in frameiter:
            result = f.copy()
            result["grid"] = board_clean(result["grid"])
            yield result

    def calc_line_clears(self, frameiter):
        prev = None
        for f in self.clean_frames(self.combine_frames(frameiter)):
            result = f.copy()
            result["lines"] = []
            if prev is not None:
                nplaced = pieces_placed(prev, result)
                assert len(which_pieces(prev, result)) == nplaced
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
                