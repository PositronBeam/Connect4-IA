import unittest

from agent import Agent
from game import Game, GameState
from model import GenRandomModel

import config

import numpy as np




class TestAgentMethods(unittest.TestCase):

    def init_random_agent(self, mcts_sims=config.MCTS_SIMS):

        result = Agent('random_agent', config.GRID_SHAPE[0] * config.GRID_SHAPE[1], config.GRID_SHAPE[1], mcts_sims, config.CPUCT, GenRandomModel())
        return result

    def test_act_game_start_1(self):
        """Empty board. Whatever PLAYER_1 plays, victory is not sure."""

        agent = self.init_random_agent()

        env = Game()
        # act(game_state, 1) : random (the agent choses randomly one action with respect to the propabilities computed by MTCS)
        # act(game_state, 0) : deterministic (the agent follows the action with the best probability computed by the MCTS)
        action, pi, MCTS_value, NN_value = agent.act(env.gameState, 1)

        self.assertEqual(NN_value, config.VALUE_DEFAULT_ACTION)
        self.assertNotEqual(MCTS_value, config.VALUE_VICTORY)

    def test_act_game_start_2(self):
        """Empty board. Whatever PLAYER_1 plays, victory is not sure,
        but a large numbers of simulations should make it understand
        that 0 or 6 are less powerful starts.
        It might also find that 3 has the best potential."""

        agent = self.init_random_agent(mcts_sims=50000)

        env = Game()
        # act(game_state, 1) : learning mode
        # act(game_state, 0) : deterministic
        action, pi, MCTS_value, NN_value = agent.act(env.gameState, 1)

        self.assertNotIn(action, [0, 6])
        self.assertEqual(action, 3)
        self.assertNotEqual(MCTS_value, config.VALUE_VICTORY)


    def test_act_game_end_1(self):
        """PLAYER_1 wins if he choses action 1, 3 or 5"""

        agent = self.init_random_agent()
        
        board = np.full(config.GRID_SHAPE, config.PLAYER_1, dtype=np.int8)
        board[-1,:] = config.NONE
        for i in range(0, config.GRID_SHAPE[1], 2):
            board[:-1,i] = config.PLAYER_2

        game_state = GameState(grid_shape = config.GRID_SHAPE, currentPlayer=config.PLAYER_1, board=board)
        # act(game_state, 1) : learning mode
        # act(game_state, 0) : deterministic
        action, pi, MCTS_value, NN_value = agent.act(game_state, 1)

        self.assertIn(action, [1, 3, 5])
        self.assertEqual(MCTS_value, config.VALUE_VICTORY)


    def test_act_game_end_2(self):
        """PLAYER_1 wins if he choses action 1"""

        agent = self.init_random_agent()
        
        board = np.full(config.GRID_SHAPE, config.NONE, dtype=np.int8)
        board[:2,:] = config.PLAYER_1
        board[2,1] = config.PLAYER_1
        for i in range(0, config.GRID_SHAPE[1], 2):
            board[:2,i] = config.PLAYER_2

        game_state = GameState(grid_shape = config.GRID_SHAPE, currentPlayer=config.PLAYER_1, board=board)
        # act(game_state, 1) : learning mode
        # act(game_state, 0) : deterministic
        action, pi, MCTS_value, NN_value = agent.act(game_state, 1)

        self.assertEqual(action, 1)
        self.assertEqual(MCTS_value, config.VALUE_VICTORY)


    def test_act_game_mid_1(self):
        """PLAYER_1 should chose action 1, 3 or 5, but not sure to win"""

        agent = self.init_random_agent(mcts_sims=5000)
        
        board = np.full(config.GRID_SHAPE, config.NONE, dtype=np.int8)
        board[:2,:] = config.PLAYER_1
        for i in range(0, config.GRID_SHAPE[1], 2):
            board[:2,i] = config.PLAYER_2

        game_state = GameState(grid_shape = config.GRID_SHAPE, currentPlayer=config.PLAYER_1, board=board)
        # act(game_state, 1) : learning mode
        # act(game_state, 0) : deterministic
        action, pi, MCTS_value, NN_value = agent.act(game_state, 1)

        self.assertIn(action, [1, 3, 5])
        self.assertNotEqual(MCTS_value, config.VALUE_VICTORY)
    


    def test_act_game_mid_2(self):
        """PLAYER_1 should chose action 2, in order to prevent victory of PLAYER_2"""

        agent = self.init_random_agent(mcts_sims=5000)
        
        board = np.full(config.GRID_SHAPE, config.NONE, dtype=np.int8)
        board[:2,:] = config.PLAYER_1
        board[2,2] = config.PLAYER_2
        for i in range(0, config.GRID_SHAPE[1], 2):
            board[:2,i] = config.PLAYER_2

        game_state = GameState(grid_shape = config.GRID_SHAPE, currentPlayer=config.PLAYER_1, board=board)
        # act(game_state, 1) : learning mode
        # act(game_state, 0) : deterministic
        action, pi, MCTS_value, NN_value = agent.act(game_state, 1)

        self.assertEqual(action, 2)
        self.assertNotEqual(MCTS_value, config.VALUE_VICTORY)


    def test_act_game_mid_3(self):
        """PLAYER_2 should chose action 4, in order to win"""
        """Because whatever PLAYER_1 does after that, PLAYER_2 wins"""

        agent = self.init_random_agent(mcts_sims=5000)
        
        board = np.full(config.GRID_SHAPE, config.NONE, dtype=np.int8)
        # X = PLAYER_1 , O = PLAYER_2
        #['-', '-', '-', '-', '-', '-', '-']
        #['-', '-', '-', '-', '-', '-', '-']
        #['-', '-', '-', 'X', '-', '-', '-']
        #['-', '-', '-', 'O', '-', '-', '-']
        #['X', 'X', '-', 'X', '-', '-', '-']
        #['X', 'O', '-', 'O', '-', 'O', '-']
        board[0,0]=board[1,0]=board[1,1]=board[1,3]=board[3,3]=config.PLAYER_1
        board[0,1]=board[0,3]=board[2,3]=board[0,5]=config.PLAYER_2

        game_state = GameState(grid_shape = config.GRID_SHAPE, currentPlayer=config.PLAYER_2, board=board)
        # act(game_state, 1) : learning mode
        # act(game_state, 0) : deterministic
        action, pi, MCTS_value, NN_value = agent.act(game_state, 1)

        self.assertEqual(action, 4)
        self.assertNotEqual(MCTS_value, config.VALUE_VICTORY)

    def test_act_game_mid_4(self):
        """column 0 is full. Check that the game cannot chose it"""

        agent = self.init_random_agent()
        
        board = np.full(config.GRID_SHAPE, config.PLAYER_1, dtype=np.int8)
        board[-1,1:] = config.NONE
        for i in range(0, config.GRID_SHAPE[1], 2):
            board[:-1,i] = config.PLAYER_2

        game_state = GameState(grid_shape = config.GRID_SHAPE, currentPlayer=config.PLAYER_1, board=board)
        # act(game_state, 1) : learning mode
        # act(game_state, 0) : deterministic
        action, pi, MCTS_value, NN_value = agent.act(game_state, 1)

        self.assertIn(action, [1, 3, 5])
        self.assertNotEqual(action, 0)
        self.assertEqual(MCTS_value, config.VALUE_VICTORY)

