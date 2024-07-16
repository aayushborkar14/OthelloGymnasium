"""Othello environments for reinforcement learning."""

import gymnasium
from gymnasium import spaces
from gymnasium.spaces import Discrete
import pygame
import numpy as np
import json


class OthelloEnv(gymnasium.Env):
    """Othello Base Env"""

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 4}

    def __init__(self,
                 protagonist=1,
                 board_size=8,
                 sudden_death_on_invalid_move=True,
                 render_in_step=False,
                 num_disk_as_reward=False,
                 possible_actions_in_obs=True,
                 render_mode=None,
                 mute=False):

        # Locals
        self.CELL_DIM = 50
        self.BACK_COLOR = (70,156,46)
        self.BLACK = (0,0,0)
        self.WHITE = (255,255,255)

        self.BLACK_DISK = -1
        self.NO_DISK = 0
        self.WHITE_DISK = 1

        # Create basic environment.
        self.board_size = board_size
        self.num_disk_as_reward = num_disk_as_reward

        self.render_in_step = render_in_step

        # Initialize policies.
        self.protagonist = protagonist

        # Initialize action space: one action for each board position.
        self.action_space = spaces.Discrete(self.board_size ** 2)

        # Initialize observation space.
        if possible_actions_in_obs:
            self.observation_space = spaces.Box(
                np.float32(-np.ones([2, ] + [self.board_size] * 2)),
                np.float32(np.ones([2, ] + [self.board_size] * 2)))
        else:
            self.observation_space = spaces.Box(
                np.float32(-np.ones([self.board_size] * 2)),
                np.float32(np.ones([self.board_size] * 2)))

        # Initializeis from configs.
        self.board_size = max(4, board_size)
        self.render_mode = render_mode
        self.sudden_death_on_invalid_move = sudden_death_on_invalid_move
        self.board_state = self._reset_board()
        self.viewer = None
        self.clock = None
        self.num_disk_as_reward = num_disk_as_reward
        self.mute = mute  # Log msgs can be misleading when planning with model.
        self.possible_actions_in_obs = possible_actions_in_obs

        # Initialize internal states.
        self.player_turn = self.BLACK_DISK
        self.winner = self.NO_DISK
        self.terminated = False
        self.possible_moves = []
        self.selected_move = None

    @property
    def get_observation(self):
        """
        Interface Property: get_observation
        
        Get state of the current Board
        """
        return self._get_obs()

    @property
    def get_player_turn(self):
        """
        Interface Property: get_player_turn
        
        Get the color who have to make the next move
        """
        return self.player_turn

    @property
    def get_terminated(self):
        """
        Interface Property: get_termninated
        
        it returns True if the current game is terminated
        """
        return self.terminated

    @property
    def get_possible_moves(self):
        """
        Interface Property: get_possible_moves
        
        it returns returns the list of possibile moves
        """
        return self.possible_moves

    @property
    def get_board_size(self):
        """
        Interface Property: get_board_size
        
        it returns returns the size of the current board
        """
        return self.board_size

    @property
    def n_players(self):
        """
        Interface Property: n_players
        
        it returns returns the number of players
        """
        return 2

    def reset(self, seed=None, options=None):
        """
        Interface Method: reset()

        Compliant to GYMNASIUM.reset()

        options parameter not implemented
        seed    parameter not implemented
        """

        self.board_state = self._reset_board()
        self.player_turn = self.BLACK_DISK
        self.winner = self.NO_DISK
        self.terminated = False
        self.possible_moves = self.get_possible_actions()
        info = self._get_info()
        return self._get_obs(), info

    def step(self, action):
        """
        Interface Method: step()

        Compliant to GYMNASIUM.step()
        """

        # Apply action.
        if self.terminated:
            raise ValueError('Game has terminated!')
        if action not in self.possible_moves:
            invalid_action = True
        else:
            invalid_action = False
        if not invalid_action:
            self.update_board(action)

        # Determine if game should terminate.
        num_vacant_positions = (self.board_state == self.NO_DISK).sum()
        no_more_vacant_places = num_vacant_positions == 0
        sudden_death = invalid_action and self.sudden_death_on_invalid_move
        done = sudden_death or no_more_vacant_places

        current_player = self.player_turn
        if done:
            # If game has terminated, determine winner.
            self.winner = self.determine_winner(sudden_death=sudden_death)
        else:
            # If game continues, determine who moves next.
            self.set_player_turn(-self.player_turn)
            if len(self.possible_moves) == 0:
                self.set_player_turn(-self.player_turn)
                if len(self.possible_moves) == 0:
                    if not self.mute:
                        print('No possible moves for either party.')
                    self.winner = self.determine_winner()

        reward = 0
        if self.terminated:
            if self.num_disk_as_reward:
                if sudden_death:
                    # Strongly discourage invalid actions.
                    reward = -(self.board_size ** 2)
                else:
                    white_cnt, black_cnt = self.count_disks()
                    if current_player == self.WHITE_DISK:
                        reward = white_cnt - black_cnt
                        if black_cnt == 0:
                            reward = self.board_size ** 2
                    else:
                        reward = black_cnt - white_cnt
                        if white_cnt == 0:
                            reward = self.board_size ** 2
            else:
                reward = self.winner * current_player

        if self.render_in_step and (not done):
            self.render()
        return self._get_obs(), reward, self.terminated, False, self._get_info()

    def render(self):
        """
        Interface Method: render()

        Compliant to GYMNASIUM.render()
        """

        if self.render_mode == 'ansi':
            return self.print_board()
        else:
            return self._render_frame()

    def close(self):
        """
        Interface Method: close()

        Compliant to GYMNASIUM.close()
        """

        if self.viewer is not None:
            pygame.display.quit()
            pygame.quit()
            self.viewer = None
        if self.render_mode == 'rgb_array':
            pygame.quit()

    def _reset_board(self):
        """
        Auxiliary Method: _reset_board()
        """

        board_state = np.zeros([self.board_size] * 2, dtype=int)
        center_row_ix = center_col_ix = self.board_size // 2
        board_state[center_row_ix - 1][center_col_ix - 1] = self.WHITE_DISK
        board_state[center_row_ix][center_col_ix] = self.WHITE_DISK
        board_state[center_row_ix][center_col_ix - 1] = self.BLACK_DISK
        board_state[center_row_ix - 1][center_col_ix] = self.BLACK_DISK
        return board_state

    def _get_obs(self):
        """
        Auxiliary Method: _get_obs()
        """

        state = self.board_state
        if self.possible_actions_in_obs:
            grid_of_possible_moves = np.zeros(self.board_size ** 2, dtype=bool)
            grid_of_possible_moves[self.possible_moves] = True
            var_temp = np.concatenate([np.expand_dims(state, axis=0),
                                   grid_of_possible_moves.reshape(
                                       [1, self.board_size, self.board_size])],
                                  axis=0)
            var_temp = var_temp.astype("float32")
            return var_temp
        else:
            state = state.astype("float32")
            return state

    def _get_info(self):
        """
        Auxiliary Method: _get_info()
        """

        white_cnt, black_cnt = self.count_disks()
        return { "white_disks": white_cnt, "black_disks": black_cnt}

    def get_num_killed_enemy(self, board, x, y, delta_x, delta_y):
        """
        Auxiliary Method: get_num_killed_enemy()
        """

        # We overload self.WHITE_DISK to be our disk
        # and self.BLACK_DISK to be enemies.
        # (x, y) is a valid position if the following pattern exists:
        # "(x, y), self.BLACK_DISK, ..., self.BLACK_DISK, self.WHITE_DISK"

        next_x = x + delta_x
        next_y = y + delta_y

        # The neighbor must be an enemy.
        if (
                next_x < 0 or
                next_x >= self.board_size or
                next_y < 0 or
                next_y >= self.board_size or
                board[next_x][next_y] != self.BLACK_DISK
        ):
            return 0

        # Keep scanning in the direction.
        cnt = 0
        while (
                0 <= next_x < self.board_size and
                0 <= next_y < self.board_size and
                board[next_x][next_y] == self.BLACK_DISK
        ):
            next_x += delta_x
            next_y += delta_y
            cnt += 1

        if (
                next_x < 0 or
                next_x >= self.board_size or
                next_y < 0 or
                next_y >= self.board_size or
                board[next_x][next_y] != self.WHITE_DISK
        ):
            return 0
        else:
            return cnt

    def get_possible_actions(self, board=None):
        """
        Auxiliary Method: get_possible_actions()
        """

        actions = []
        if board is None:
            if self.player_turn == self.WHITE_DISK:
                board = self.board_state
            else:
                board = -self.board_state

        for row_ix in range(self.board_size):
            for col_ix in range(self.board_size):
                if board[row_ix][col_ix] == self.NO_DISK:
                    if (
                            self.get_num_killed_enemy(
                                board, row_ix, col_ix, 1, 1) or
                            self.get_num_killed_enemy(
                                board, row_ix, col_ix, 1, 0) or
                            self.get_num_killed_enemy(
                                board, row_ix, col_ix, 1, -1) or
                            self.get_num_killed_enemy(
                                board, row_ix, col_ix, 0, 1) or
                            self.get_num_killed_enemy(
                                board, row_ix, col_ix, 0, -1) or
                            self.get_num_killed_enemy(
                                board, row_ix, col_ix, -1, 1) or
                            self.get_num_killed_enemy(
                                board, row_ix, col_ix, -1, 0) or
                            self.get_num_killed_enemy(
                                board, row_ix, col_ix, -1, -1)
                    ):
                        actions.append(row_ix * self.board_size + col_ix)
        return actions

    def print_board(self, print_valid_moves=True):
        """
        Auxiliary Method: print_board()
        """

        valid_actions = self.get_possible_actions()

        if print_valid_moves:
            board = self.board_state.copy().ravel()
            for p in valid_actions:
                board[p] = int(p)+10

            board = board.reshape(*self.board_state.shape)
        else:
            board = self.board_state

        strBoard = 'Turn: {}'.format(
            'WHITE' if self.player_turn == self.WHITE_DISK else 'BLACK')

        strBoard += '\nValid actions: {}'.format(valid_actions)

        for row in board:
            strTmp = "\n"
            for el in row:
                if el == -1:
                    strTmp += " BB ".rjust(5)
                elif el == 0:
                    strTmp += " OO ".rjust(5)
                elif el == 1:
                    strTmp += " WW ".rjust(5)
                else:
                    strTmp += ("-" + str(el - 10).rjust(2) + "-").rjust(5)
            strBoard += strTmp

        strBoard += '\n' + '-' * 10
        print(strBoard)
        return strBoard

    def set_board_state(self, board_state, perspective=1):
        """
        Auxiliary Method: set_board_state()

        Force setting the board state, necessary in model-based RL

        """
        if np.ndim(board_state) > 2:
            state = board_state[0]
        else:
            state = board_state
        if perspective == self.WHITE_DISK:
            self.board_state = np.array(state)
        else:
            self.board_state = -np.array(state)

    def update_board(self, action):
        """
        Auxiliary Method: update_board()
        """

        x = action // self.board_size
        y = action % self.board_size

        if self.player_turn == self.BLACK_DISK:
            self.board_state = -self.board_state

        for delta_x in [-1, 0, 1]:
            for delta_y in [-1, 0, 1]:
                if not (delta_x == 0 and delta_y == 0):
                    kill_cnt = self.get_num_killed_enemy(
                        self.board_state, x, y, delta_x, delta_y)
                    for i in range(kill_cnt):
                        dx = (i + 1) * delta_x
                        dy = (i + 1) * delta_y
                        self.board_state[x + dx][y + dy] = self.WHITE_DISK
        self.board_state[x][y] = self.WHITE_DISK

        if self.player_turn == self.BLACK_DISK:
            self.board_state = -self.board_state

    def set_player_turn(self, turn):
        """
        Auxiliary Method: set_player_turn()
        """

        self.player_turn = turn
        self.possible_moves = self.get_possible_actions()

    def count_disks(self):
        """
        Auxiliary Method: count_disks()
        """

        white_cnt = (self.board_state == self.WHITE_DISK).sum()
        black_cnt = (self.board_state == self.BLACK_DISK).sum()
        return white_cnt, black_cnt

    def determine_winner(self, sudden_death=False):
        """
        Auxiliary Method: determine_winner()
        """

        self.terminated = True
        if sudden_death:
            if not self.mute:
                print('sudden death due to rule violation')
            if self.player_turn == self.WHITE_DISK:
                if not self.mute:
                    print('BLACK wins')
                return self.BLACK_DISK
            else:
                if not self.mute:
                    print('WHITE wins')
                return self.WHITE_DISK
        else:
            white_cnt, black_cnt = self.count_disks()
            if not self.mute:
                print('white: {}, black: {}'.format(white_cnt, black_cnt))
            if white_cnt > black_cnt:
                if not self.mute:
                    print('WHITE wins')
                return self.WHITE_DISK
            elif black_cnt > white_cnt:
                if not self.mute:
                    print('BLACK wins')
                return self.BLACK_DISK
            else:
                if not self.mute:
                    print('DRAW')
                return self.NO_DISK

    def create_window(self):
        """
        PYGAME auxiliary function: create_window()
        """

        # Set up the drawing window
        screen = pygame.display.set_mode([self.CELL_DIM*(self.board_size),
                                          self.CELL_DIM*(self.board_size)])
        # Fill the background
        screen.fill(self.WHITE)
        #
        pygame.draw.rect(screen,
                         self.BACK_COLOR,
                         pygame.Rect(self.CELL_DIM,
                                     self.CELL_DIM,
                                     self.board_size*self.CELL_DIM,
                                     self.board_size*self.CELL_DIM))

        return screen

    def draw_board(self, screen):
        """
        PYGAME auxiliary function: draw_board(screen)
        """

        size = self.CELL_DIM*self.board_size
        canvas = pygame.Surface((size, size))
        # Fill the background
        screen.fill(self.BLACK)
        
        pygame.draw.rect(screen,
                         self.BACK_COLOR,
                         pygame.Rect(0,
                                     0,
                                     self.board_size*self.CELL_DIM,
                                     self.board_size*self.CELL_DIM))
        for j in range(0, self.board_size):
            for i in range(0, self.board_size):
                # Draw rect
                pygame.draw.rect(screen,
                                 self.BLACK,
                                 pygame.Rect(i*self.CELL_DIM,
                                             j*self.CELL_DIM,
                                             (i+1)*self.CELL_DIM,
                                             (j+1)*self.CELL_DIM),
                                             2)
        return

    def draw_frame(self, screen):
        """
        PYGAME auxiliary function: draw_frame(screen)
        """

        pygame.draw.rect(screen,
                         self.WHITE,
                         pygame.Rect((self.board_size+1)*self.CELL_DIM,
                                     0,
                                     (self.board_size+2)*self.CELL_DIM,
                                     (self.board_size+2)*self.CELL_DIM))
        pygame.draw.rect(screen,
                         self.WHITE,
                         pygame.Rect(0,
                                     (self.board_size+1)*self.CELL_DIM,
                                     (self.board_size+2)*self.CELL_DIM,
                                     (self.board_size+2)*self.CELL_DIM))
        return

    def draw_token(self, screen, x,y,color):
        """
        PYGAME auxiliary function: draw_token(screen, x, y, color)
        """

        dim_x = x + 1
        dim_y = y + 1

        pygame.draw.circle(screen,
                           color,
                           (dim_x*self.CELL_DIM-self.CELL_DIM/2,
                            dim_y*self.CELL_DIM-self.CELL_DIM/2),
                            self.CELL_DIM/2- self.CELL_DIM/10)
        return

    def draw_position(self, screen, x,y,color, text):
        """
        PYGAME auxiliary function: draw_position(screen, x, y, color, text)
        """

        dim_x = x + 1
        dim_y = y + 1
        font = pygame.font.SysFont('Arial', 12)
        screen.blit(font.render(text, True, self.BLACK),
                       (dim_x*self.CELL_DIM-2*self.CELL_DIM/3,
                        dim_y*self.CELL_DIM-2*self.CELL_DIM/3))
        pygame.draw.circle(screen,
                           color,
                           (dim_x*self.CELL_DIM-self.CELL_DIM/2,
                            dim_y*self.CELL_DIM-self.CELL_DIM/2),
                            self.CELL_DIM/2- self.CELL_DIM/10, 2)
        return

    def _render_frame(self):
        """
        PYGAME auxiliary function: _render_frame()
        """

        if self.render_mode == 'rgb_array':
            pygame.init()
            canvas = pygame.Surface((self.CELL_DIM*(self.board_size) + 2,
                                     self.CELL_DIM*(self.board_size) + 2))
            self.draw_board(canvas)
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if self.board_state[i][j] == self.WHITE_DISK:
                        self.draw_token(canvas, j,i, self.WHITE)
                    elif self.board_state[i][j] == self.BLACK_DISK:
                        self.draw_token(canvas, j,i, self.BLACK)

            color = None
            if self.player_turn == self.WHITE_DISK:
                color = self.WHITE
            else:
                color = self.BLACK

            for p in self.possible_moves:
                i = p // self.board_size
                j = p % self.board_size

                self.draw_position(canvas, j,i, color, str(i * self.board_size + j))
            self.draw_frame(canvas)

            return np.transpose(np.array(
                      pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        elif self.render_mode == 'human':

            if self.viewer is None:
                pygame.init()
                pygame.display.init()
                self.viewer = self.create_window()

            if self.clock is None:
                self.clock = pygame.time.Clock()

            # basic font for user typed
            base_font = pygame.font.Font(None, 32)

            active = False

            self.draw_board(self.viewer)

            for i in range(self.board_size):
                for j in range(self.board_size):
                    if self.board_state[i][j] == self.WHITE_DISK:
                        self.draw_token(self.viewer, j,i, self.WHITE)
                    elif self.board_state[i][j] == self.BLACK_DISK:
                        self.draw_token(self.viewer, j,i, self.BLACK)

            color = None
            if self.player_turn == self.WHITE_DISK:
                color = self.WHITE
            else:
                color = self.BLACK

            for p in self.possible_moves:
                i = p // self.board_size
                j = p % self.board_size

                self.draw_position(self.viewer, j,i, color, str(i * self.board_size + j))
                
            self.draw_frame(self.viewer)

            self.viewer.blit(self.viewer, self.viewer.get_rect())
            self.clock.tick(self.metadata["render_fps"])

            pygame.display.flip()

            self.viewer.blit(self.viewer, self.viewer.get_rect())
            pygame.event.pump()
            pygame.display.update()

