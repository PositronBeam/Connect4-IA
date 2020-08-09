
from game import Game, GameState
from model import Residual_CNN, GenRandomModel, VALUE_HEAD, POLICY_HEAD

from memory import Memory
from agent import Agent

import loggers as lg
import config
import play_matches
import pickle
import random

CURRENT_PLAYER_NAME = 'current_player'
BEST_PLAYER_NAME = 'best_player'

lg.logger_main.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')
lg.logger_main.info('=*=*=*=*=*=.      NEW LOG      =*=*=*=*=*')
lg.logger_main.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')


memory = Memory(config.MEMORY_SIZE)

# The idea is as follows:
# player1 has to play. His Monte Carlo Tree Search does N simulations in order to evaluate the best possible move.
# If N is very big, like 5000+, the estimation should be quite accurate. But if N is smaller, like 50, the estimate
# will be wrong because the MCTS will stop after 2 or 3 moves maximum, and the "expectations" at the leaves corresponding
# to these game states will not be estimated correctly.
# This is where the neural networks arrives: if trained on a big number of states (with correct expectations),
# it will be able to predict correctly the state of the leaves of the MCTS, and the global estimation of the MCTS
# will be much better.
# As the beginning, the predictions of the neural network will be wrong, so the results of the MCTS will be wrong as well,
# but after enough games, both the neural network and the MCTS  will improve and converge.
env = Game()

# create an untrained neural network objects from the config file
current_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (1,) + config.GRID_SHAPE,   config.GRID_SHAPE[1], config.HIDDEN_CNN_LAYERS)
current_NN.compile_with_loss_weights(loss_weights={VALUE_HEAD: 0.0, POLICY_HEAD: 1.0})
best_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (1,) +  config.GRID_SHAPE,   config.GRID_SHAPE[1], config.HIDDEN_CNN_LAYERS)
best_NN.compile_with_loss_weights(loss_weights={VALUE_HEAD: 0.0, POLICY_HEAD: 1.0})
best_NN.model.set_weights(current_NN.model.get_weights())

# Au début, les réseaux de neurones font n'importe quoi, du coup j'essaie un truc plus déterministe: un MCTS qui fait 5000 simulations par coup
# Les pi (probas des coups) seront de bonne qualité, mais les action value (estimation de la récompense) seront toujours à 0
# Du coup au 1er training, je répartis le poids des heads des réseaux de neurones uniquement sur les pi
# Ensuite, je vide la mémoire pour vider les AV (ça vide aussi les pi, tant pis) et je remets des poids à 0.5 partout pour les heads
current_player = Agent(CURRENT_PLAYER_NAME, config.GRID_SHAPE[0] * config.GRID_SHAPE[1], config.GRID_SHAPE[1], 5000, config.CPUCT, GenRandomModel())
best_player = Agent(BEST_PLAYER_NAME, config.GRID_SHAPE[0] * config.GRID_SHAPE[1], config.GRID_SHAPE[1], 5000, config.CPUCT, GenRandomModel())

best_player_version = 0

iteration = 0

while 1:

    iteration += 1
    
    lg.logger_main.info('ITERATION NUMBER ' + str(iteration))
    
    lg.logger_main.info('BEST PLAYER VERSION: %d', best_player_version)

    ######## SELF PLAY ########
    lg.logger_main.info('SELF PLAYING ' + str(config.EPISODES) + ' EPISODES...')
    _, memory, _, _ = play_matches.playMatches(current_player, best_player, config.EPISODES, lg.logger_main, turns_until_tau0 = config.TURNS_UNTIL_TAU0, memory = memory)
    
    memory.clear_stmemory()
    
    if len(memory.ltmemory) >= config.MEMORY_SIZE:

        ######## RETRAINING ########
        if iteration == 1:
            current_player.model = current_NN
            best_player.model = best_NN
            current_player.MCTSsimulations = config.MCTS_SIMS
            best_player.MCTSsimulations = config.MCTS_SIMS
        else:
            best_NN.compile_with_loss_weights(loss_weights={VALUE_HEAD: 0.5, POLICY_HEAD: 0.5})
            current_NN.compile_with_loss_weights(loss_weights={VALUE_HEAD: 0.5, POLICY_HEAD: 0.5})
        lg.logger_main.info('RETRAINING...')
        current_player.replay(memory.ltmemory)

        if iteration == 1: 
            pickle.dump( memory, open( config.run_folder + "memory/memory" + str(iteration).zfill(4) + ".p", "wb" ) )
            memory.clear_ltmemory()
            
        if iteration % 5 == 0:
            pickle.dump( memory, open( config.run_folder + "memory/memory" + str(iteration).zfill(4) + ".p", "wb" ) )

            
        ######## TOURNAMENT ########
        lg.logger_main.info('TOURNAMENT...')
        scores, _, points, sp_scores = play_matches.playMatches(best_player, current_player, config.EVAL_EPISODES, lg.logger_tourney, turns_until_tau0 = 0, memory = None)
        
        lg.logger_main.info('SCORES')
        lg.logger_main.info(scores)
        lg.logger_main.info('STARTING PLAYER / NON-STARTING PLAYER SCORES')
        lg.logger_main.info(sp_scores)
        

        if scores[CURRENT_PLAYER_NAME] > scores[BEST_PLAYER_NAME] * config.SCORING_THRESHOLD:
            best_player_version = best_player_version + 1
            best_NN.model.set_weights(current_NN.model.get_weights())
            best_NN.write(best_player_version)

    else:
        lg.logger_main.info('MEMORY SIZE: ' + str(len(memory.ltmemory)))