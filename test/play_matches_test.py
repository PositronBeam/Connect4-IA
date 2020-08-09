
import unittest

import config
import play_matches
import loggers as lg
import logging

from memory import Memory
from model import GenRandomModel, Residual_CNN
from agent import Agent

class TestPlayMatches(unittest.TestCase):

    def test_play_matches_fake_neural_network(self):

        memory = Memory(config.MEMORY_SIZE)

        player1 = Agent('random_agent_1', config.GRID_SHAPE[0] * config.GRID_SHAPE[1], config.GRID_SHAPE[1], config.MCTS_SIMS, config.CPUCT, GenRandomModel())
        player2 = Agent('random_agent_2', config.GRID_SHAPE[0] * config.GRID_SHAPE[1], config.GRID_SHAPE[1], config.MCTS_SIMS, config.CPUCT, GenRandomModel())
        
        logger = lg.logger_main
        logger.setLevel(logging.DEBUG)
        scores, memory, points, sp_scores = play_matches.playMatches(player1, player2, config.EPISODES, logger, turns_until_tau0 = config.TURNS_UNTIL_TAU0, memory = memory)



    def test_play_matches_neural_network(self):
    
        memory = Memory(config.MEMORY_SIZE)

        # At the beginning, we set a random model. It will be similar to an untrained CNN, and quicker.
        # We also set config.MCTS_SIMS, which is rather low, and will produce poor estimations from the MCTS.
        # The idea is encourage exploration and generate a lot of boards in memory, even if the probabilities
        # associated to their possible actions are wrong.
        # Memory is completed at the end of the game according to the final winner, in order to correct the values
        # of each move. All the moves of the winner receive value=1 and all the moves of the loser receive value=-1
        # The neural network will learn to predict the probabilities and the values.
        # It will learn wrong probas and values at the beginning, but after some time, the CNN and the neural network
        # will improve from eachother and converge.
        player1 = Agent('cnn_agent_1', config.GRID_SHAPE[0] * config.GRID_SHAPE[1], config.GRID_SHAPE[1], config.MCTS_SIMS, config.CPUCT, GenRandomModel())
        player2 = Agent('cnn_agent_2', config.GRID_SHAPE[0] * config.GRID_SHAPE[1], config.GRID_SHAPE[1], config.MCTS_SIMS, config.CPUCT, GenRandomModel())
        
        scores, memory, points, sp_scores = play_matches.playMatches(player1, player2, config.EPISODES, lg.logger_main, turns_until_tau0 = config.TURNS_UNTIL_TAU0, memory = memory)

        # play_matches.playMatches() has copied stmemory to ltmemory, so we can clear stmemory safely
        memory.clear_stmemory()

        cnn1 = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (1,) + config.GRID_SHAPE, config.GRID_SHAPE[1], config.HIDDEN_CNN_LAYERS)
        cnn2 = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (1,) + config.GRID_SHAPE, config.GRID_SHAPE[1], config.HIDDEN_CNN_LAYERS)
        cnn2.model.set_weights(cnn1.model.get_weights())
        cnn1.plot_model()

        player1.model = cnn1

        ######## RETRAINING ########
        player1.replay(memory.ltmemory)

        for _ in range(1):

            scores, memory, points, sp_scores = play_matches.playMatches(player1, player2, config.EPISODES, lg.logger_main, turns_until_tau0 = config.TURNS_UNTIL_TAU0, memory = memory)

            # play_matches.playMatches() has copied stmemory to ltmemory, so we can clear stmemory safely
            memory.clear_stmemory()

            player1.replay(memory.ltmemory)

        
        print('TOURNAMENT...')
        scores, _, points, sp_scores = play_matches.playMatches(player1, player2, config.EVAL_EPISODES, lg.logger_main, turns_until_tau0 = 0, memory = None)
        print('\nSCORES')
        print(scores)
        print('\nSTARTING PLAYER / NON-STARTING PLAYER SCORES')
        print(sp_scores)


