--- START OF FILE evalminimax.py ---

# THE INVERTED ARENA: Minimax (First Player) vs. Our AlphaZero Master (Second Player).
# The definitive experiment.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import math

# --- BATTLE CONFIGURATION ---
MODEL_PATH = "checkers_master_final.pth"
MINIMAX_DEPTH = 8
MCTS_SIMS_FOR_MASTER = 200

# --- GAME AND AI DEFINITIONS ---
BOARD_SIZE = 8
DEVICE = torch.device("cpu")

# --- Game Logic, Neural Network, MCTS (Same as previous scripts) ---
class Checkers:
    def get_initial_board(self):
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8);
        for r in range(3):
            for c in range(BOARD_SIZE):
                if (r + c) % 2 == 1: board[r, c] = -1 # Black pieces
        for r in range(5, BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if (r + c) % 2 == 1: board[r, c] = 1 # White pieces
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

def state_to_tensor(board, player):
    tensor = np.zeros((5, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    tensor[0, board == player] = 1; tensor[1, board == player*2] = 1
    tensor[2, board == -player] = 1; tensor[3, board == -player*2] = 1
    if player == 1: tensor[4,:,:] = 1.0
    return torch.from_numpy(tensor).unsqueeze(0).to(DEVICE)

class PolicyValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        num_channels = 64
        self.body = nn.Sequential(nn.Conv2d(5, num_channels, 3, padding=1), nn.BatchNorm2d(num_channels), nn.ReLU(),
                                  nn.Conv2d(num_channels, num_channels, 3, padding=1), nn.BatchNorm2d(num_channels), nn.ReLU(),
                                  nn.Conv2d(num_channels, num_channels, 3, padding=1), nn.BatchNorm2d(num_channels), nn.ReLU())
        self.policy_head = nn.Sequential(nn.Conv2d(num_channels, 4, 1), nn.BatchNorm2d(4), nn.ReLU(), nn.Flatten(),
                                         nn.Linear(4 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE))
        self.value_head = nn.Sequential(nn.Conv2d(num_channels, 2, 1), nn.BatchNorm2d(2), nn.ReLU(), nn.Flatten(),
                                        nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, 64), nn.ReLU(),
                                        nn.Linear(64, 1), nn.Tanh())
    def forward(self, x):
        x = self.body(x); return self.policy_head(x), self.value_head(x)

class MCTSNode:
    def __init__(self, parent=None, prior=0.0):
        self.parent = parent; self.prior = prior; self.children = {}; self.visits = 0; self.value_sum = 0.0
    def get_value(self): return self.value_sum / self.visits if self.visits > 0 else 0.0

class MCTS:
    def __init__(self, game, model, sims=100, c_puct=1.5):
        self.game, self.model, self.sims, self.c_puct = game, model, sims, c_puct
    def run(self, board, player):
        root = MCTSNode()
        self._expand_and_evaluate(root, board, player)
        for _ in range(self.sims):
            node, search_board, search_player = root, np.copy(board), player
            search_path = [root]
            while node.children:
                move, node = self._select_child(node)
                search_board = self.game.apply_move(search_board, move); search_player *= -1; search_path.append(node)
            value = self.game.check_game_over(search_board, search_player)
            if value is None and node.visits == 0: value = self._expand_and_evaluate(node, search_board, search_player)
            elif value is None: value = node.get_value()
            for n in reversed(search_path): n.visits += 1; n.value_sum += value; value *= -1
        moves = list(root.children.keys())
        visits = np.array([root.children[m].visits for m in moves])
        return moves, visits / (np.sum(visits) + 1e-9)
    def _select_child(self, node):
        sqrt_total_visits = np.sqrt(node.visits); best_move, max_score = None, -np.inf
        for move, child in node.children.items():
            score = -child.get_value() + self.c_puct * child.prior * sqrt_total_visits / (1 + child.visits)
            if score > max_score: max_score, best_move = score, move
        return best_move, node.children[best_move]
    def _expand_and_evaluate(self, node, board, player):
        valid_moves = self.game.get_valid_moves(board, player)
        if not valid_moves: return -1.0
        with torch.no_grad():
            policy_logits, value_tensor = self.model(state_to_tensor(board, player))
        value = value_tensor.item()
        policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
        move_priors = {}; total_prior = 0
        for move in valid_moves:
            if isinstance(move, list): start_pos_tuple = move[0][0]
            else: start_pos_tuple = move[0]
            start_pos_idx = start_pos_tuple[0] * BOARD_SIZE + start_pos_tuple[1]
            prior = policy_probs[start_pos_idx]
            key = tuple(move) if isinstance(move, list) else move
            move_priors[key] = prior; total_prior += prior
        if total_prior > 0:
            for move_key, prior in move_priors.items(): node.children[move_key] = MCTSNode(parent=node, prior=prior / total_prior)
        else:
            for move in valid_moves:
                key = tuple(move) if isinstance(move, list) else move
                node.children[key] = MCTSNode(parent=node, prior=1.0 / len(valid_moves))
        return value

# --- MINIMAX BRAIN ---
def evaluate_board(board, player):
    score = 0; pawn_value = 1; king_value = 2.5
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            piece = board[r, c]
            if piece * player > 0: # Our pieces
                score += king_value if abs(piece) == 2 else pawn_value
                if abs(piece) == 1: score += (BOARD_SIZE - 1 - r) * 0.1 if player == 1 else r * 0.1
            elif piece * player < 0: # Opponent pieces
                score -= king_value if abs(piece) == 2 else pawn_value
    return score

def minimax_alpha_beta(board, depth, alpha, beta, maximizing_player, game_logic):
    current_turn_player = 1 if maximizing_player else -1
    game_over = game_logic.check_game_over(board, current_turn_player)
    if depth == 0 or game_over is not None:
        if game_over is not None:
            # The score is always from the perspective of the maximizing player (player 1, 'x')
            if game_over == 1: return math.inf  # Player 1 won
            if game_over == -1: return -math.inf # Player -1 won (loss for Player 1)
        return evaluate_board(board, 1) # Always evaluate from Player 1's perspective

    valid_moves = game_logic.get_valid_moves(board, current_turn_player)
    
    if maximizing_player:
        max_eval = -math.inf
        for move in valid_moves:
            new_board = game_logic.apply_move(board, move)
            eval = minimax_alpha_beta(new_board, depth - 1, alpha, beta, False, game_logic)
            max_eval = max(max_eval, eval); alpha = max(alpha, eval)
            if beta <= alpha: break
        return max_eval
    else: # Minimizing player
        min_eval = math.inf
        for move in valid_moves:
            new_board = game_logic.apply_move(board, move)
            eval = minimax_alpha_beta(new_board, depth - 1, alpha, beta, True, game_logic)
            min_eval = min(min_eval, eval); beta = min(beta, eval)
            if beta <= alpha: break
        return min_eval

# --- PLAYERS ---
class AIPlayer:
    def __init__(self, model_path, game_logic, mcts_sims):
        self.game = game_logic; self.model = PolicyValueNetwork().to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE)); self.model.eval()
        self.mcts = MCTS(self.game, self.model, sims=mcts_sims)
        print("AI Player (AlphaZero Master) loaded and ready.")
    def get_move(self, board, player):
        print("AlphaZero Master is thinking..."); start_time = time.time()
        valid_moves, policy = self.mcts.run(np.copy(board), player)
        move = valid_moves[np.argmax(policy)]
        print(f"(MCTS Time: {time.time() - start_time:.2f}s)"); return move

class MinimaxPlayer:
    def __init__(self, depth, game_logic):
        self.depth = depth; self.game = game_logic
        print(f"Minimax Player (depth {depth}) ready.")
    def get_move(self, board, player):
        print(f"Minimax (depth {self.depth}) is thinking..."); start_time = time.time()
        best_move = None
        best_value = -math.inf if player == 1 else math.inf # Adjust for maximizing or minimizing
        
        valid_moves = self.game.get_valid_moves(board, player)
        
        for i, move in enumerate(valid_moves):
            print(f"  Evaluating move {i+1}/{len(valid_moves)}...", end='\r')
            new_board = self.game.apply_move(board, move)
            board_value = minimax_alpha_beta(new_board, self.depth - 1, -math.inf, math.inf, player == -1, self.game)
            if player == 1: # Maximizing
                if board_value > best_value: best_value, best_move = board_value, move
            else: # Minimizing
                if board_value < best_value: best_value, best_move = board_value, move
        print(f"\n(Minimax Time: {time.time() - start_time:.2f}s)"); return best_move

# --- THE FINAL ARENA (INVERTED) ---
def play_match(model_path, minimax_depth):
    game = Checkers()
    
    # ✅ INVERSION: Minimax plays first with white pieces ('x', player 1)
    player_minimax = MinimaxPlayer(minimax_depth, game)
    # ✅ INVERSION: Our Master plays second with black pieces ('o', player -1)
    player_az = AIPlayer(model_path, game, MCTS_SIMS_FOR_MASTER)

    board = game.get_initial_board()
    current_player_val = 1
    
    move_count = 0
    while move_count < 150:
        result = game.check_game_over(board, current_player_val)
        if result is not None: break
            
        print("\n" + "="*40)
        chars = {1: 'x', 2: 'X', -1: 'o', -2: 'O', 0: '.'};
        print("  0 1 2 3 4 5 6 7 (cols)");
        for r_idx, row in enumerate(board): print(f"{r_idx} {' '.join(chars[val] for val in row)}")
        print("="*40)

        # ✅ INVERTED TURN LOGIC
        if current_player_val == 1:
            move = player_minimax.get_move(board, current_player_val)
            print(f"\nMINIMAX PLAYS (x): {move}")
        else:
            move = player_az.get_move(board, current_player_val)
            print(f"\nALPHAZERO MASTER PLAYS (o): {move}")
        
        if move is None:
            print("A player could not choose a move. Game over.")
            result = 1 if current_player_val == -1 else -1
            break
            
        board = game.apply_move(board, move)
        current_player_val *= -1
        move_count += 1
    
    print("\n" + "#"*40); print("### GAME OVER ###"); print("#"*40)
    if result == -1:
        winner = "MINIMAX" if -current_player_val == 1 else "ALPHAZERO MASTER"
        print(f"THE WINNER IS: {winner}!")
    elif result == 1:
        winner = "MINIMAX" if current_player_val == 1 else "ALPHAZERO MASTER"
        print(f"THE WINNER IS: {winner}!")
    else: print("DRAW!")

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file '{MODEL_PATH}' not found.")
    else:
        play_match(MODEL_PATH, MINIMAX_DEPTH)