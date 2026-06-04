"""Minimal Connect 4 rules for in-process self-play.

Board is a (6,7) int8 array, row 0 = bottom, values 0 empty / 1 first / 2
second — matching the wire encoding the model server decodes, so the planes a
self-play net trains on are identical to what it sees when served.
"""

from __future__ import annotations

import numpy as np

ROWS, COLS = 6, 7
_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


class Board:
    __slots__ = ("cells", "ply")

    def __init__(self, cells: np.ndarray | None = None, ply: int = 0) -> None:
        self.cells = cells if cells is not None else np.zeros((ROWS, COLS), dtype=np.int8)
        self.ply = ply

    def copy(self) -> "Board":
        return Board(self.cells.copy(), self.ply)

    @property
    def to_move(self) -> int:
        return 1 if self.ply % 2 == 0 else 2

    def legal(self) -> list[int]:
        return [c for c in range(COLS) if self.cells[ROWS - 1, c] == 0]

    def play(self, col: int) -> "Board":
        nxt = self.copy()
        piece = nxt.to_move
        for r in range(ROWS):
            if nxt.cells[r, col] == 0:
                nxt.cells[r, col] = piece
                break
        nxt.ply += 1
        return nxt

    def winner(self) -> int:
        c = self.cells
        for r in range(ROWS):
            for col in range(COLS):
                p = c[r, col]
                if p == 0:
                    continue
                for dr, dc in _DIRS:
                    rr, cc, n = r, col, 0
                    while 0 <= rr < ROWS and 0 <= cc < COLS and c[rr, cc] == p:
                        n += 1
                        if n == 4:
                            return int(p)
                        rr += dr
                        cc += dc
        return 0

    def terminal(self) -> tuple[bool, float]:
        """(is_terminal, value for the side to move). Loss = -1, draw = 0."""
        if self.winner() != 0:
            return True, -1.0  # opponent just completed four; side to move lost
        if self.ply >= ROWS * COLS:
            return True, 0.0
        return False, 0.0

    def planes(self) -> np.ndarray:
        """2×6×7 side-to-move planes (own stones, opponent stones)."""
        mover = self.to_move
        own = (self.cells == mover).astype(np.float32)
        opp = ((self.cells != 0) & (self.cells != mover)).astype(np.float32)
        return np.stack([own, opp], axis=0)
