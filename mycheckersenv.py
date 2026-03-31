import functools
from typing import Dict, List, Optional, Tuple

import gymnasium
import numpy as np
from gymnasium.spaces import Box, Discrete, Dict as GymDict
from gymnasium.utils import seeding

from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers


BOARD_SIZE = 6
ACTION_DIRECTIONS = 8
MAX_ACTIONS = BOARD_SIZE * BOARD_SIZE * ACTION_DIRECTIONS
MAX_MOVES_PER_EPISODE = 200

EMPTY = 0
P1_MAN = 1
P1_KING = 2
P2_MAN = -1
P2_KING = -2


def env(render_mode: Optional[str] = None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    created_env = raw_env(render_mode=internal_render_mode)
    if render_mode == "ansi":
        created_env = wrappers.CaptureStdoutWrapper(created_env)
    created_env = wrappers.AssertOutOfBoundsWrapper(created_env)
    created_env = wrappers.OrderEnforcingWrapper(created_env)
    return created_env


def raw_env(render_mode: Optional[str] = None):
    return CheckersAECEnv(render_mode=render_mode)


class CheckersAECEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "checkers_6x6_v0"}

    def __init__(self, render_mode: Optional[str] = None):
        self.possible_agents = ["player_0", "player_1"]
        self.agent_name_mapping = {"player_0": 0, "player_1": 1}
        self.render_mode = render_mode
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.move_count = 0
        self.np_random = None
        self.np_random_seed = None
        self._agent_selector = None
        self._last_observation_masks = {}
        self.must_continue_jump: Optional[Tuple[int, int]] = None

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(MAX_ACTIONS)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        board_space = Box(low=0, high=1, shape=(BOARD_SIZE, BOARD_SIZE, 4), dtype=np.int8)
        mask_space = Box(low=0, high=1, shape=(MAX_ACTIONS,), dtype=np.int8)
        return GymDict({"board": board_space, "action_mask": mask_space})

    def reset(self, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)
        self.agents = self.possible_agents[:]
        self.rewards = {"player_0": 0.0, "player_1": 0.0}
        self._cumulative_rewards = {"player_0": 0.0, "player_1": 0.0}
        self.terminations = {"player_0": False, "player_1": False}
        self.truncations = {"player_0": False, "player_1": False}
        self.infos = {"player_0": {}, "player_1": {}}
        self.move_count = 0
        self.must_continue_jump = None
        self._setup_initial_board()
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self._refresh_masks_for_all_agents()

    def _setup_initial_board(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        row_index = 0
        while row_index < 2:
            col_index = 0
            while col_index < BOARD_SIZE:
                if (row_index + col_index) % 2 == 1:
                    self.board[row_index, col_index] = P2_MAN
                col_index += 1
            row_index += 1
        row_index = BOARD_SIZE - 2
        while row_index < BOARD_SIZE:
            col_index = 0
            while col_index < BOARD_SIZE:
                if (row_index + col_index) % 2 == 1:
                    self.board[row_index, col_index] = P1_MAN
                col_index += 1
            row_index += 1

    def observe(self, agent):
        board_tensor = self._encode_board_for_agent(agent)
        action_mask = self._last_observation_masks[agent]
        return {"board": board_tensor, "action_mask": action_mask}

    def _encode_board_for_agent(self, agent):
        player_is_zero = agent == "player_0"
        board_tensor = np.zeros((BOARD_SIZE, BOARD_SIZE, 4), dtype=np.int8)
        row_index = 0
        while row_index < BOARD_SIZE:
            col_index = 0
            while col_index < BOARD_SIZE:
                value = self.board[row_index, col_index]
                if player_is_zero:
                    if value == P1_MAN:
                        board_tensor[row_index, col_index, 0] = 1
                    elif value == P1_KING:
                        board_tensor[row_index, col_index, 1] = 1
                    elif value == P2_MAN:
                        board_tensor[row_index, col_index, 2] = 1
                    elif value == P2_KING:
                        board_tensor[row_index, col_index, 3] = 1
                else:
                    if value == P2_MAN:
                        board_tensor[row_index, col_index, 0] = 1
                    elif value == P2_KING:
                        board_tensor[row_index, col_index, 1] = 1
                    elif value == P1_MAN:
                        board_tensor[row_index, col_index, 2] = 1
                    elif value == P1_KING:
                        board_tensor[row_index, col_index, 3] = 1
                col_index += 1
            row_index += 1
        return board_tensor

    def step(self, action: Optional[int]):
        current_agent = self.agent_selection
        if self.terminations[current_agent] or self.truncations[current_agent]:
            self._was_dead_step(action)
            return

        self._cumulative_rewards[current_agent] = 0.0
        legal_actions = self._legal_action_indices_for_agent(current_agent)
        # If an agent proposes an action that isn't legal, we end the game immediately.
        if action is None or action not in legal_actions:
            self._apply_illegal_move_penalty(current_agent)
            self._accumulate_rewards()
            return

        from_row, from_col, to_row, to_col, jumped_cell = self._decode_and_validate_action(action, current_agent)
        self._execute_move(from_row, from_col, to_row, to_col, jumped_cell)
        self.move_count += 1

        self._clear_rewards()

        winner_no_pieces = self._winner_if_side_has_no_pieces()
        if winner_no_pieces is not None:
            self.terminations = {"player_0": True, "player_1": True}
            self.must_continue_jump = None
            self._apply_winner_rewards(winner_no_pieces)
            self._accumulate_rewards()
            return

        if jumped_cell is not None:
            player_is_zero = current_agent == "player_0"
            landing_piece = int(self.board[to_row, to_col])
            further_captures = self._capture_actions_only(
                to_row, to_col, landing_piece, player_is_zero
            )
            if len(further_captures) > 0:
                # Multi-jump rule: the same player keeps moving with the same piece.
                self.must_continue_jump = (to_row, to_col)
                self._refresh_masks_for_all_agents()
                self._accumulate_rewards()
                if self.render_mode == "human":
                    self.render()
                return

        self.must_continue_jump = None

        if self.move_count >= MAX_MOVES_PER_EPISODE:
            self.must_continue_jump = None
            self.truncations = {"player_0": True, "player_1": True}
            self._clear_rewards()
            self._accumulate_rewards()
            return

        next_agent = self._other_agent(current_agent)
        if len(self._legal_action_indices_for_agent(next_agent)) == 0:
            self.terminations = {"player_0": True, "player_1": True}
            self._apply_winner_rewards(current_agent)
            self._accumulate_rewards()
            return

        self.agent_selection = next_agent
        self._refresh_masks_for_all_agents()
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def _decode_and_validate_action(self, action, agent):
        # Action encoding: from_square * 8 + direction_code.
        from_square = action // ACTION_DIRECTIONS
        direction_code = action % ACTION_DIRECTIONS
        from_row = from_square // BOARD_SIZE
        from_col = from_square % BOARD_SIZE

        # First 4 codes are one-step diagonals, last 4 codes are the two-step jumps.
        direction_vectors = [
            (-1, -1), (-1, 1), (1, -1), (1, 1),
            (-2, -2), (-2, 2), (2, -2), (2, 2),
        ]
        row_delta, col_delta = direction_vectors[direction_code]
        to_row = from_row + row_delta
        to_col = from_col + col_delta
        jumped_cell = None
        if abs(row_delta) == 2:
            jumped_cell = (from_row + (row_delta // 2), from_col + (col_delta // 2))
        return from_row, from_col, to_row, to_col, jumped_cell

    def _execute_move(self, from_row, from_col, to_row, to_col, jumped_cell):
        piece = self.board[from_row, from_col]
        self.board[from_row, from_col] = EMPTY
        self.board[to_row, to_col] = piece
        if jumped_cell is not None:
            jumped_row, jumped_col = jumped_cell
            self.board[jumped_row, jumped_col] = EMPTY
        self._promote_if_needed(to_row, to_col)

    def _promote_if_needed(self, row_index, col_index):
        piece = self.board[row_index, col_index]
        if piece == P1_MAN and row_index == 0:
            self.board[row_index, col_index] = P1_KING
        elif piece == P2_MAN and row_index == BOARD_SIZE - 1:
            self.board[row_index, col_index] = P2_KING

    def _refresh_masks_for_all_agents(self):
        masks = {}
        for agent in self.possible_agents:
            mask = np.zeros(MAX_ACTIONS, dtype=np.int8)
            legal_actions = self._legal_action_indices_for_agent(agent)
            for legal_action in legal_actions:
                mask[legal_action] = 1
            masks[agent] = mask
        self._last_observation_masks = masks

    def _legal_action_indices_for_agent(self, agent):
        player_is_zero = agent == "player_0"
        if self.must_continue_jump is not None:
            row_index, col_index = self.must_continue_jump
            if not self._is_cell_on_board(row_index, col_index):
                return []
            current_piece = self.board[row_index, col_index]
            if not self._piece_belongs_to_agent(current_piece, player_is_zero):
                return []
            # During a forced multi-jump, only capture moves from the landing square are allowed.
            piece_value = int(current_piece)
            _, piece_captures = self._moves_for_piece(
                row_index, col_index, piece_value, player_is_zero
            )
            return piece_captures

        legal_capture_moves = []
        legal_normal_moves = []
        row_index = 0
        while row_index < BOARD_SIZE:
            col_index = 0
            while col_index < BOARD_SIZE:
                current_piece = self.board[row_index, col_index]
                if self._piece_belongs_to_agent(current_piece, player_is_zero):
                    piece_moves, piece_captures = self._moves_for_piece(row_index, col_index, current_piece, player_is_zero)
                    legal_normal_moves.extend(piece_moves)
                    legal_capture_moves.extend(piece_captures)
                col_index += 1
            row_index += 1
        if len(legal_capture_moves) > 0:
            return legal_capture_moves
        return legal_normal_moves

    def _capture_actions_only(
        self, row_index: int, col_index: int, piece: int, player_is_zero: bool
    ) -> List[int]:
        _, capture_list = self._moves_for_piece(
            row_index, col_index, piece, player_is_zero
        )
        return capture_list

    def _winner_if_side_has_no_pieces(self) -> Optional[str]:
        # Win by elimination: if a side has zero remaining pieces, the other side wins.
        player_zero_count = 0
        player_one_count = 0
        row_index = 0
        while row_index < BOARD_SIZE:
            col_index = 0
            while col_index < BOARD_SIZE:
                piece = self.board[row_index, col_index]
                if piece == P1_MAN or piece == P1_KING:
                    player_zero_count += 1
                elif piece == P2_MAN or piece == P2_KING:
                    player_one_count += 1
                col_index += 1
            row_index += 1
        if player_zero_count == 0:
            return "player_1"
        if player_one_count == 0:
            return "player_0"
        return None

    def _moves_for_piece(self, row_index, col_index, piece, player_is_zero):
        allowed_directions = self._allowed_directions_for_piece(piece, player_is_zero)
        normal_actions = []
        capture_actions = []
        for direction in allowed_directions:
            row_delta, col_delta = direction
            to_row = row_index + row_delta
            to_col = col_index + col_delta
            if self._is_cell_on_board(to_row, to_col) and self.board[to_row, to_col] == EMPTY:
                # Normal move: land on the next diagonal empty square.
                action_index = self._encode_action(row_index, col_index, row_delta, col_delta)
                normal_actions.append(action_index)

            jump_row = row_index + (2 * row_delta)
            jump_col = col_index + (2 * col_delta)
            mid_row = row_index + row_delta
            mid_col = col_index + col_delta
            if self._is_cell_on_board(jump_row, jump_col) and self.board[jump_row, jump_col] == EMPTY:
                jumped_piece = self.board[mid_row, mid_col]
                if self._is_opponent_piece(jumped_piece, player_is_zero):
                    # Capture move: jump over an opponent piece into an empty landing square.
                    jump_action = self._encode_action(row_index, col_index, 2 * row_delta, 2 * col_delta)
                    capture_actions.append(jump_action)
        return normal_actions, capture_actions

    def _allowed_directions_for_piece(self, piece, player_is_zero):
        if abs(piece) == 2:
            return [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        if player_is_zero:
            return [(-1, -1), (-1, 1)]
        return [(1, -1), (1, 1)]

    def _encode_action(self, from_row, from_col, row_delta, col_delta):
        from_square = from_row * BOARD_SIZE + from_col
        direction_to_code = {
            (-1, -1): 0, (-1, 1): 1, (1, -1): 2, (1, 1): 3,
            (-2, -2): 4, (-2, 2): 5, (2, -2): 6, (2, 2): 7,
        }
        direction_code = direction_to_code[(row_delta, col_delta)]
        return from_square * ACTION_DIRECTIONS + direction_code

    def _piece_belongs_to_agent(self, piece, player_is_zero):
        if player_is_zero:
            return piece == P1_MAN or piece == P1_KING
        return piece == P2_MAN or piece == P2_KING

    def _is_opponent_piece(self, piece, player_is_zero):
        if player_is_zero:
            return piece == P2_MAN or piece == P2_KING
        return piece == P1_MAN or piece == P1_KING

    def _is_cell_on_board(self, row_index, col_index):
        row_valid = row_index >= 0 and row_index < BOARD_SIZE
        col_valid = col_index >= 0 and col_index < BOARD_SIZE
        return row_valid and col_valid

    def _apply_illegal_move_penalty(self, acting_agent):
        other_agent = self._other_agent(acting_agent)
        self.rewards[acting_agent] = -1.0
        self.rewards[other_agent] = 1.0
        self.terminations = {"player_0": True, "player_1": True}
        self.must_continue_jump = None
        self._refresh_masks_for_all_agents()

    def _apply_winner_rewards(self, winner):
        loser = self._other_agent(winner)
        self.rewards[winner] = 1.0
        self.rewards[loser] = -1.0

    def _other_agent(self, agent):
        if agent == "player_0":
            return "player_1"
        return "player_0"

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn("Render was called without render_mode.")
            return
        symbol_map = {
            EMPTY: ".",
            P1_MAN: "x",
            P1_KING: "X",
            P2_MAN: "o",
            P2_KING: "O",
        }
        print("\n  0 1 2 3 4 5")
        row_index = 0
        while row_index < BOARD_SIZE:
            row_cells = []
            col_index = 0
            while col_index < BOARD_SIZE:
                row_cells.append(symbol_map[int(self.board[row_index, col_index])])
                col_index += 1
            print(str(row_index) + " " + " ".join(row_cells))
            row_index += 1
        print("Current agent:", self.agent_selection)
        print("Move count:", self.move_count)
        if self.must_continue_jump is not None:
            print("Continue jump from:", self.must_continue_jump)

    def close(self):
        return
