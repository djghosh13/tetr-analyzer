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


# Piece analysis

class PieceSolver:
    def __init__(self):
        self.piecelist = ""

    def _all_pieces(self, framelist):
        pieces = ""
        for frame in framelist:
            nextps = frame["next"]
            for k in range(5, -1, -1):
                if pieces.endswith(nextps[:k]):
                    pieces = pieces + nextps[k:]
                    break
        pieces = list(set("TSZIOLJ") - set(pieces[:6]))[0] + pieces
        if not all(set("TSZIOLJ") == set(pieces[i:i + 7]) for i in range(0, len(pieces) - 7, 7)):
            _log("Invalid pieces:", "".join(pieces), file=sys.stderr)
        return pieces

    def _count_pieces(self, framelist):
        idx = 0
        for frame in framelist:
            idx += self.piecelist[idx:].index(frame["next"])
            frame["PSolver"]["index"] = idx - 1

    def _num_placed(self, framelist):
        pidx, phold = None, None
        for frame in framelist:
            idx, hold = frame["PSolver"]["index"], frame["hold"]
            frame["PSolver"]["count"] = 0
            if pidx is not None:
                frame["PSolver"]["count"] = idx - pidx
                if phold == "-" and hold != "-":
                    frame["PSolver"]["count"] -= 1
                if frame["PSolver"]["count"] > 1:
                    print("Uh oh")
            pidx, phold = idx, hold

    def _current_piece(self, framelist):
        pidx, phold = None, None
        possible = []
        for frame in framelist:
            idx, hold = frame["PSolver"]["index"], frame["hold"]
            if pidx is not None:
                if hold != phold:
                    if phold != "-":
                        possible.append(phold)
                possible += self.piecelist[pidx + 1:idx + 1]
                if hold != phold:
                    possible.remove(hold)
                possible = possible[frame["PSolver"]["count"]:]
            else:
                possible += self.piecelist[0]
            assert len(possible) == 1
            frame["PSolver"]["current"] = possible[0]
            pidx, phold = idx, hold

    def _which_pieces(self, framelist):
        pidx, pcurr, phold = None, None, None
        for frame in framelist:
            idx, curr, hold = frame["PSolver"]["index"], frame["PSolver"]["current"], frame["hold"]
            frame["PSolver"]["which"] = []
            if pidx is not None:
                if phold != "-":
                    frame["PSolver"]["which"].append(phold)
                frame["PSolver"]["which"].append(pcurr)
                frame["PSolver"]["which"].extend(self.piecelist[pidx + 1:idx + 1])
                frame["PSolver"]["which"].remove(curr)
                if hold != "-":
                    frame["PSolver"]["which"].remove(hold)
            pidx, pcurr, phold = idx, curr, hold
    
    def compute(self, framelist):
        for frame in framelist:
            frame["PSolver"] = {}
        self.piecelist = self._all_pieces(framelist)
        self._count_pieces(framelist)
        self._num_placed(framelist)
        self._current_piece(framelist)
        self._which_pieces(framelist)
        return framelist


# Utilities

def to_array(grid):
    return grid.copy() if isinstance(grid, np.ndarray) else np.array([list(row) for row in grid], dtype=np.object)

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
    "S": to_filter(["** ", " **"]),
    "Z": to_filter([" **", "** "]),
    "T": to_filter(["***", " * "]),
    "I": to_filter(["****"]),
    "L": to_filter(["***", "  *"]),
    "J": to_filter(["***", "*  "]),
    "O": to_filter(["**", "**"])
}

def pprint_board(board):
    print("\n".join("".join(row) for row in board[::-1]))
    print()

# Board analysis

class BoardSolver:
    def __init__(self):
        pass

    def _clean_board(self, framelist):
        for frame in framelist:
            board = to_array(frame["grid"])[::-1]
            # Separate garbage
            garbage = []
            while len(board) and np.sum(board[0] == "G") == GRID_WIDTH - 1:
                garbage.append(board[0])
                board = board[1:]
            frame["BSolver"]["garbage"] = to_array(garbage)
            # Remove floating pieces
            for ii in range(len(board)):
                if np.all(board[ii] == "-"):
                    board = board[:ii]
                    break
            frame["BSolver"]["upstack"] = board
            frame["BSolver"]["board"] = to_array(
                garbage +
                list(board) +
                ["-" * GRID_WIDTH] * (GRID_HEIGHT - len(garbage) - len(board))
            )

    def _apply_placement(self, board, piece, x, y, rotation):
        newb = to_array(board)
        ftr = _piece_filters[piece][rotation]
        Hf, Wf = ftr.shape
        newb[y:y + Hf, x:x + Wf][ftr] = piece
        # Check for line clears
        cleared = np.nonzero((newb != "-").all(axis=-1))[0]
        newb = to_array(
            [newb[i] for i in range(len(newb)) if i not in cleared] +
            ["-" * GRID_WIDTH] * len(cleared)
        )
        return newb

    def _grid_states(self, board, piece):
        rotations = [2, 2, 4, 2, 4, 4, 1]["SZTILJO".index(piece)]
        filled = (board != "-")
        for rot in range(rotations):
            ftr = _piece_filters[piece][rot]
            Hf, Wf = ftr.shape
            for y in range(board.shape[0] - Hf + 1):
                for x in range(board.shape[1] - Wf + 1):
                    if filled[y:y + Hf, x:x + Wf][ftr].any():
                        continue
                    if y > 0 and not filled[y - 1, x:x + Wf].any():
                        continue
                    # Valid placement, simulate placement
                    yield self._apply_placement(board, piece, x, y, rot)

    def _equal(self, boardA, boardB):
        garbdiff = np.sum(boardB == "G") - np.sum(boardA == "G")
        if garbdiff < 0 or garbdiff % (GRID_WIDTH - 1):
            return False
        garbdiff //= GRID_WIDTH - 1
        boardA, boardB = boardA[:len(boardA) - garbdiff], boardB[garbdiff:]
        return np.all(boardA == boardB)

    def _almost_equal(self, boardA, boardB):
        garbdiff = np.sum(boardB == "G") - np.sum(boardA == "G")
        if garbdiff < 0 or garbdiff % (GRID_WIDTH - 1):
            return False
        garbdiff //= GRID_WIDTH - 1
        boardA, boardB = boardA[:len(boardA) - garbdiff], boardB[garbdiff:]
        return (np.all(boardA[boardA != "-"] == boardB[boardA != "-"]) and
            np.all(boardA[boardB == "-"] == "-"))
    
    def _similarity(self, boardA, boardB):
        garbdiff = np.sum(boardB == "G") - np.sum(boardA == "G")
        if garbdiff < 0 or garbdiff % (GRID_WIDTH - 1):
            return -np.inf
        garbdiff //= GRID_WIDTH - 1
        boardA, boardB = boardA[:len(boardA) - garbdiff], boardB[garbdiff:]
        if (np.all(boardA[boardA != "-"] == boardB[boardA != "-"]) and
            np.all(boardA[boardB == "-"] == "-")):
            return (boardA != boardB).sum()
        retun -np.inf

    def _match_height(self, boardA, boardB):
        garbdiff = (np.sum(boardB == "G") - np.sum(boardA == "G")) // 9
        boardA, boardB = boardA[:len(boardA) - garbdiff], boardB[garbdiff:]
        return to_array((list(boardB[:garbdiff]) + list(boardA))[:len(boardA)])

    def _track_board(self, framelist):
        board = np.empty([GRID_HEIGHT, GRID_WIDTH], dtype=np.object)
        board[...] = "-"
        for frame in framelist:
            if not frame["PSolver"]["count"]:
                # No pieces placed, intersect board
                try:
                    assert np.all(board[board != "-"] == frame["BSolver"]["board"][board != "-"])
                except AssertionError as e:
                    pprint_board(board)
                    pprint_board(frame["grid"][::-1])
                    raise e
            else:
                if frame["PSolver"]["count"] == 1:
                    # Piece placed, check lines cleared
                    changed = False
                    piece = frame["PSolver"]["which"][0]
                    for newboard in self._grid_states(board, piece):
                        if self._almost_equal(newboard, frame["BSolver"]["board"]):
                            print(f'Placed {piece}')
                            board = self._match_height(newboard, frame["BSolver"]["board"])
                            pprint_board(board)
                            changed = True
                            break
                    if not changed:
                        print(f'Frame {frame["frame"]}: Could not locate piece: {piece}')
                        pprint_board(board)
                        pprint_board(frame["BSolver"]["board"])
                else:
                    print("Uh oh too many pieces")
                    board = frame["BSolver"]["board"]

    def compute(self, framelist):
        for frame in framelist:
            frame["BSolver"] = {}
        self._clean_board(framelist)
        self._track_board(framelist)
        return framelist
        