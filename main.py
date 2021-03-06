
from game import Game, GameState
from model import Residual_CNN
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

# See "https://github.com/tensorflow/tensorflow/issues/24828": without that part of code, I got an error:
#"tensorflow.python.framework.errors_impl.UnknownError:  Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above."

import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU') 
for physical_device in physical_devices: 
    tf.config.experimental.set_memory_growth(physical_device, True)


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
best_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (1,) +  config.GRID_SHAPE,   config.GRID_SHAPE[1], config.HIDDEN_CNN_LAYERS)
best_NN.model.set_weights(current_NN.model.get_weights())

current_player = Agent(CURRENT_PLAYER_NAME, config.GRID_SHAPE[0] * config.GRID_SHAPE[1], config.GRID_SHAPE[1], config.MCTS_SIMS, config.CPUCT, current_NN)
best_player = Agent(BEST_PLAYER_NAME, config.GRID_SHAPE[0] * config.GRID_SHAPE[1], config.GRID_SHAPE[1], config.MCTS_SIMS, config.CPUCT, best_NN)

best_player_version = 0

iteration = 0

while 1:

    iteration += 1
    
    lg.logger_main.info('ITERATION NUMBER ' + str(iteration))
    
    lg.logger_main.info('BEST PLAYER VERSION: %d', best_player_version)

    ######## SELF PLAY ########
    lg.logger_main.info('SELF PLAYING ' + str(config.EPISODES) + ' EPISODES...')
    _, memory, _, _ = play_matches.playMatches(best_player, best_player, config.EPISODES, lg.logger_main, turns_until_tau0 = config.TURNS_UNTIL_TAU0, memory = memory)
    
    memory.clear_stmemory()
    
    if len(memory.ltmemory) >= config.MEMORY_SIZE:

        ######## RETRAINING ########
        lg.logger_main.info('RETRAINING...')
        current_player.replay(memory.ltmemory)

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