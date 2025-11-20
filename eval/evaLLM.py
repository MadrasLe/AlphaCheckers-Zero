--- START OF FILE evaLLM.py ---

# BATTLE ARENA SCRIPT: Checkers Master (AlphaZero) vs. Large Language Model (LLM via Groq).

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import random
from groq import Groq

# --------------------------------------------------------------------------
# --- STEP 1: ADVERSARY SETUP (LLM) ---
# --------------------------------------------------------------------------

# ✅ PLACE YOUR GROQ API KEY HERE
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY_HERE"

# Initialize the API client
try:
    client = Groq()
    print("Groq client initialized successfully.")
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    print("Make sure the GROQ_API_KEY environment variable is set correctly.")
    exit()

# --------------------------------------------------------------------------
# --- STEP 2: MASTER'S CODE (LOGIC AND MODEL) ---
# --------------------------------------------------------------------------
# All definitions of the Checkers Master must be here for it to function.

BOARD_SIZE = 8
DEVICE = torch.device("cpu")

# --- Game Logic, Neural Network, MCTS (Copied from previous script) ---
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

# --------------------------------------------------------------------------
# --- STEP 3: TRANSLATOR AND PLAYER CLASSES ---
# --------------------------------------------------------------------------

def board_to_text(board, player_to_move):
    """Converts the NumPy board to a text representation friendly for the LLM."""
    chars = {1: 'x', 2: 'X', -1: 'o', -2: 'O', 0: '.'}
    s = "  0 1 2 3 4 5 6 7 (cols)\n"
    for r_idx, row in enumerate(board):
        s += f"{r_idx} {' '.join(chars[val] for val in row)}\n"
    
    player_char = 'x (lowercase)' if player_to_move == 1 else 'o (lowercase)'
    s += f"\nIt is your turn. You are player '{player_char}'."
    return s

def text_to_move(response_text, valid_moves):
    """Tries to extract the move index from the LLM response."""
    try:
        # Looks for a number in the response
        numbers = [int(s) for s in response_text.split() if s.isdigit()]
        if numbers:
            chosen_idx = numbers[0]
            if 0 <= chosen_idx < len(valid_moves):
                return valid_moves[chosen_idx]
    except (ValueError, IndexError):
        pass
    print("[WARNING] LLM did not return a valid index. Trying again...")
    return None

class AIPlayer:
    """Our AlphaZero Master."""
    def __init__(self, model_path, game_logic, mcts_sims=200):
        self.game = game_logic
        self.model = PolicyValueNetwork().to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
        self.mcts = MCTS(self.game, self.model, sims=mcts_sims)
        print("AI Player (AlphaZero Master) loaded and ready.")

    def get_move(self, board, player):
        print("AlphaZero Master is thinking...")
        valid_moves, policy = self.mcts.run(np.copy(board), player)
        move = valid_moves[np.argmax(policy)]
        return move

class LLMPlayer:
    """The LLM Adversary."""
    def __init__(self, groq_client):
        self.client = groq_client
        self.model_name = "moonshotai/kimi-k2-instruct-0905"
        print(f"LLM Player (using {self.model_name}) ready.")

    def get_move(self, board, player, valid_moves):
        print("LLM is thinking...")
        
        board_str = board_to_text(board, player)
        moves_str = "Your valid moves are:\n"
        for i, move in enumerate(valid_moves):
            moves_str += f"  {i}: {move}\n"

        system_prompt = """You are a world-class Checkers player. Your goal is to analyze the board and choose the best possible move from the provided list. Checkers rules apply, including mandatory captures. You must answer ONLY with the index number of the move you chose. Your response must be JUST the number, nothing else."""
        
        user_prompt = f"{board_str}\n{moves_str}\nChoose your move number:"
        
        for _ in range(3): # Try up to 3 times in case of invalid response
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=10,
            )
            response = completion.choices[0].message.content
            print(f"LLM answered: '{response.strip()}'")
            move = text_to_move(response, valid_moves)
            if move:
                return move
            time.sleep(2) # Wait a bit before trying again
        
        print("[ERROR] LLM failed to provide a valid move after 3 attempts. Choosing a random move.")
        return random.choice(valid_moves)

# --------------------------------------------------------------------------
# --- STEP 4: THE ARENA ---
# --------------------------------------------------------------------------

def play_match(model_path):
    game = Checkers()
    
    # Our Master will play with white pieces (1)
    player_az = AIPlayer(model_path, game)
    
    # The LLM will play with black pieces (-1)
    player_llm = LLMPlayer(client)

    board = game.get_initial_board()
    current_player_val = 1
    
    move_count = 0
    while move_count < 150:
        result = game.check_game_over(board, current_player_val)
        if result is not None:
            break
            
        print("\n" + "="*40)
        # Print the board
        chars = {1: 'x', 2: 'X', -1: 'o', -2: 'O', 0: '.'}
        print("  0 1 2 3 4 5 6 7 (cols)")
        for r_idx, row in enumerate(board):
            print(f"{r_idx} {' '.join(chars[val] for val in row)}")
        print("="*40)

        valid_moves = game.get_valid_moves(board, current_player_val)
        
        if current_player_val == 1:
            move = player_az.get_move(board, current_player_val)
            print(f"\nMASTER ALPHAZERO PLAYS (x): {move}")
        else:
            move = player_llm.get_move(board, current_player_val, valid_moves)
            print(f"\nLLM PLAYS (o): {move}")
        
        board = game.apply_move(board, move)
        current_player_val *= -1
        move_count += 1
    
    print("\n" + "#"*40)
    print("### GAME OVER ###")
    print("#"*40)
    
    # The result is from the perspective of the LAST player to move, who lost because they had no moves.
    # So if result is -1 (loss), the winner is the PREVIOUS player.
    if result == -1:
        winner = -current_player_val
        if winner == 1:
            print("MASTER ALPHAZERO WON!")
        else:
            print("THE LLM WON!")
    elif result == 1:
        winner = current_player_val
        if winner == 1:
            print("MASTER ALPHAZERO WON!")
        else:
            print("THE LLM WON!")
    else: # Draw by move limit
        print("DRAW!")

if __name__ == "__main__":
    model_file = "checkers_master_final.pth"
    if not os.path.exists(model_file):
        print(f"ERROR: Model file '{model_file}' not found.")
    else:
        play_match(model_file)