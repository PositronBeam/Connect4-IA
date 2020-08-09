import numpy as np
import random
import logging

import loggers as lg

from game import Game, GameState
from model import Residual_CNN

from agent import Agent, User

import config

def playMatchesBetweenVersions(env, run_version, player1version, player2version, EPISODES, logger, turns_until_tau0, goes_first = 0):
    
    if player1version == -1:
        player1 = User('player1', env.state_size, env.action_size)
    else:
        player1_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, env.input_shape,   env.action_size, config.HIDDEN_CNN_LAYERS)

        if player1version > 0:
            player1_network = player1_NN.read(env.name, run_version, player1version)
            player1_NN.model.set_weights(player1_network.get_weights())   
        player1 = Agent('player1', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, player1_NN)

    if player2version == -1:
        player2 = User('player2', env.state_size, env.action_size)
    else:
        player2_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, env.input_shape,   env.action_size, config.HIDDEN_CNN_LAYERS)
        
        if player2version > 0:
            player2_network = player2_NN.read(env.name, run_version, player2version)
            player2_NN.model.set_weights(player2_network.get_weights())
        player2 = Agent('player2', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, player2_NN)

    scores, memory, points, sp_scores = playMatches(player1, player2, EPISODES, logger, turns_until_tau0, None, goes_first)

    return (scores, memory, points, sp_scores)


def playMatches(player1, player2, EPISODES, logger, turns_until_tau0, memory = None):

    env = Game()
    scores = {player1.name:0, "drawn": 0, player2.name:0}
    sp_scores = {'sp':0, "drawn": 0, 'nsp':0}
    points = {player1.name:[], player2.name:[]}

    for e in range(EPISODES):

        logger.debug('====================')
        logger.debug('EPISODE %d OF %d', e+1, EPISODES)
        logger.debug('====================')

        state = env.reset()
        
        done = 0
        turn = 0
        player1.mcts = None
        player2.mcts = None

        players = {config.PLAYER_1: player1, config.PLAYER_2: player2}
        logger.debug(player1.name + ' plays as ' + config.RENDER_PLAYERS[config.PLAYER_1])
        logger.debug('--------------')

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(env.gameState.render())

        while not done:
            turn = turn + 1
    
            #### Run the MCTS algo and return an action
            deterministic = 1 if turn < turns_until_tau0 else 0
            action, pi, MCTS_value, NN_value = players[state.currentPlayer].act(state, deterministic)

            if memory != None:
                ####Commit the move to memory
                memory.commit_stmemory(state, pi)


            logger.debug('action: %d', action)
            logger.debug(['{0:.2f}'.format(np.round(x,2)) for x in pi])
            logger.debug('MCTS perceived value for %s: %f', config.RENDER_PLAYERS[state.currentPlayer] ,np.round(MCTS_value,2))
            logger.debug('NN perceived value for %s: %f', config.RENDER_PLAYERS[state.currentPlayer] ,np.round(NN_value,2))
            logger.debug('====================')

            ### Do the action
            state, value, done = env.step(action) #the value of the newState from the POV of the new playerTurn i.e. -1 if the previous player played a winning move
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(env.gameState.render())

            if done: 
                # The last player just made his move, env.step() switched to the other player and detected that the game was finished
                # For example, player1 moved, he won, and env.step() switched to player2 and detected the victory of player1: state.currentPlayer contains player2
                # => the last_player is -state.currentPlayer
                last_player = -state.currentPlayer

                if memory != None:
                    #### If the game is finished, assign the values correctly to the game moves
                    for move in memory.stmemory:
                        if move['state'].currentPlayer == last_player:
                            move['value'] = value
                        else:
                            move['value'] = -value
                         
                    memory.commit_ltmemory()
             
                if value == -config.VALUE_VICTORY:
                    logger.info('%s WINS!', players[last_player].name)
                    scores[players[last_player].name] = scores[players[last_player].name] + 1
                    if last_player == config.PLAYER_1: 
                        sp_scores['sp'] = sp_scores['sp'] + 1
                    else:
                        sp_scores['nsp'] = sp_scores['nsp'] + 1

                    points[players[last_player].name].append(config.POINTS_VICTORY)
                    points[players[-last_player].name].append(config.POINTS_DEFEAT)
                    
                else:
                    logger.info('DRAW...')
                    scores['drawn'] = scores['drawn'] + 1
                    sp_scores['drawn'] = sp_scores['drawn'] + 1


    return (scores, memory, points, sp_scores)
