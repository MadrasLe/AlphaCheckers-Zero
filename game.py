import numpy as np

BOARD_SIZE = 8

class Checkers:
    def get_initial_board(self):
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8);
        for r in range(3):
            for c in range(BOARD_SIZE):
                if (r + c) % 2 == 1: board[r, c] = -1
        for r in range(5, BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if (r + c) % 2 == 1: board[r, c] = 1
        return board
    def get_valid_moves(self, board, player):
        jumps = self._get_all_jumps(board, player)
        if jumps: return jumps
        moves = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r, c] * player > 0: moves.extend(self._get_simple_moves(board, r, c))
        return moves
    def _get_simple_moves(self, board, r, c):
        moves = []; piece = board[r, c]; player = np.sign(piece)
        directions = [(-1, -1), (-1, 1)] if player == 1 else [(1, -1), (1, 1)]
        if abs(piece) == 2: directions.extend([(1, -1), (1, 1)] if player == 1 else [(-1, -1), (-1, 1)])
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == 0: moves.append(((r, c), (nr, nc)))
        return moves
    def _get_all_jumps(self, board, player):
        all_jumps = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r, c] * player > 0:
                    jumps = self._find_jump_sequences(np.copy(board), r, c)
                    if jumps: all_jumps.extend(jumps)
        if not all_jumps: return []
        max_len = max(len(j) for j in all_jumps)
        return [j for j in all_jumps if len(j) == max_len]
    def _find_jump_sequences(self, board, r, c, path=[]):
        piece = board[r, c]; player = np.sign(piece)
        if piece == 0: return []
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)] if abs(piece) == 2 else \
                     [(-1, -1), (-1, 1)] if player == 1 else [(1, -1), (1, 1)]
        found_jumps = []
        for dr, dc in directions:
            mid_r, mid_c = r + dr, c + dc; end_r, end_c = r + 2*dr, c + 2*dc
            if 0 <= end_r < BOARD_SIZE and 0 <= end_c < BOARD_SIZE and \
               board[mid_r, mid_c] * player < 0 and board[end_r, end_c] == 0:
                move = ((r, c), (end_r, end_c))
                new_board = np.copy(board); new_board[end_r, end_c] = piece; new_board[r, c] = 0; new_board[mid_r, mid_c] = 0
                next_jumps = self._find_jump_sequences(new_board, end_r, end_c, path + [move])
                if next_jumps: found_jumps.extend(next_jumps)
                else: found_jumps.append(path + [move])
        return found_jumps
    def apply_move(self, board, move):
        b_ = np.copy(board)
        is_jump_chain = isinstance(move, list) or (isinstance(move, tuple) and isinstance(move[0], tuple) and isinstance(move[0][0], tuple))
        sub_moves = move if is_jump_chain else [move]
        for (r1, c1), (r2, c2) in sub_moves:
            piece = b_[r1, c1]
            if piece == 0: continue
            b_[r2, c2] = piece; b_[r1, c1] = 0
            if abs(r1 - r2) == 2: b_[(r1+r2)//2, (c1+c2)//2] = 0
        r_final, c_final = sub_moves[-1][1]; p_final = b_[r_final, c_final]
        if p_final == 1 and r_final == 0: b_[r_final, c_final] = 2
        if p_final == -1 and r_final == BOARD_SIZE-1: b_[r_final, c_final] = -2
        return b_
    def check_game_over(self, board, player):
        if not self.get_valid_moves(board, player): return -1
        if not np.any(np.sign(board) == -player): return 1
        return None
