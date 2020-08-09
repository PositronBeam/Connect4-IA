# %matplotlib inline

import numpy as np
import random

import MCTS as mc
from game import GameState
from model import VALUE_HEAD, POLICY_HEAD

import config
import loggers as lg
import logging
import time



class User():
	def __init__(self, name, state_size, action_size):
		self.name = name
		self.state_size = state_size
		self.action_size = action_size

	def act(self, state, tau):
		action = input('Enter your chosen action: ')
		pi = np.zeros(self.action_size)
		pi[action] = 1
		value = None
		NN_value = None
		return (action, pi, value, NN_value)



class Agent():
	def __init__(self, name, state_size, action_size, mcts_simulations, cpuct, model):
		self.name = name

		self.state_size = state_size
		self.action_size = action_size

		self.cpuct = cpuct

		self.MCTSsimulations = mcts_simulations
		self.model = model

		self.mcts = None

		self.train_overall_loss = []
		self.train_value_loss = []
		self.train_policy_loss = []
		self.val_overall_loss = []
		self.val_value_loss = []
		self.val_policy_loss = []

	
	def simulate(self):

		if lg.logger_mcts.isEnabledFor(logging.DEBUG):
			state = GameState.from_id(self.mcts.root.state_id, config.GRID_SHAPE)
			lg.logger_mcts.debug('ROOT NODE...%s', self.mcts.root.state_id)			
			lg.logger_mcts.debug(state.render())
			lg.logger_mcts.debug('CURRENT PLAYER...%d', state.currentPlayer)

		##### MOVE THE LEAF NODE
		leaf, value, done, breadcrumbs = self.mcts.moveToLeaf()
		if lg.logger_mcts.isEnabledFor(logging.DEBUG):
			state = GameState.from_id(leaf.state_id, config.GRID_SHAPE)
			lg.logger_mcts.debug(state.render())

		##### EVALUATE THE LEAF NODE
		value, breadcrumbs = self.evaluateLeaf(leaf, value, done, breadcrumbs)

		##### BACKFILL THE VALUE THROUGH THE TREE
		self.mcts.backFill(leaf, value, breadcrumbs)


	def act(self, state, tau):
		"""if tau = 1: the agent choses randomly one action with respect to the propabilities computed by MTCS
		if tau = 0: the agent follows the action with the best probability computed by the MCTS
		Example: the agent has 7 possible actions. After simulating all of them, the MCTS finds the following probabilities
		of victory: action0 = 1% , action1 = 98% , action3 = 1%, action4 = 0%... (the sum of the probabilities must be 100%)
		If tau = 0, the agent will always chose action1 because it is the highest.
		If tau = 1, the agent will have 1% of probability to chose action1, 98% to chose action2, 1% to to chose action3.
		
		When tau=1, the idea is to encourage some exploration: if the agent always follows the MCTS, the games during training
		will always be the same (or almost the same), and the agent will not have seen enough different possibilities."""

		if self.mcts == None or state.id not in self.mcts.tree:
			self.buildMCTS(state)
		else:
			self.changeRootMCTS(state)

		#### run the simulation
		for sim in range(self.MCTSsimulations):
			lg.logger_mcts.debug('***************************')
			lg.logger_mcts.debug('****** SIMULATION %d ******', sim + 1)
			lg.logger_mcts.debug('***************************')
			self.simulate()

		#### get action values
		pi, values = self.getAV(1)

		####pick the action
		action, value = self.chooseAction(pi, values, tau)

		nextState, _, _ = state.takeAction(action)

		NN_value = -self.get_preds(nextState)[0]

		lg.logger_mcts.info('ACTION VALUES...%s', pi)
		lg.logger_mcts.info('CHOSEN ACTION...%d', action)
		lg.logger_mcts.info('MCTS PERCEIVED VALUE...%f', value)
		lg.logger_mcts.info('NN PERCEIVED VALUE...%f', NN_value)

		return (action, pi, value, NN_value)


	def get_preds(self, state):
		# state.board has shape (6,7), so it can be considered as a 1-layer image of shape:
		# - either (1,6,7) if channel layer first as in model.py
		# - or (6,7,1) if channel last as usually done with Keras
		# Plus Keras needs a batch of inputs => we add an additional encapsulating array
		# => resulting shape is (1,1,6,7)
		inputToModel = np.array([[state.get_board_for_neural_network()]], dtype=np.int8)

		preds = self.model.predict(inputToModel)
		# preds[0] is an array of shape (1,1): the input was a batch of 1 board, and the neural network
		# predicts one value per board, between -1 and 1 because of the tanh activation for this head
		value = preds[0][0,0]
		# preds[1] is an array of shape (1,7): the input was a batch of 1 board, and the neural network
		# predicts 7 values per board (the values for each possible action - more precisely a linear value
		# before transformation to a percentage via the softmax)
		logits = preds[1][0]

		# Forbidden actions must receive a probability equal to 0, therefore we force the output of the
		# neural network to -100 for them (so that the softmax would transform them to 0)
		allowedActions = state.allowedActions()
		forbiddenActions = [not(isallowed) for isallowed in allowedActions]
		logits[forbiddenActions] = -100

		#SOFTMAX
		odds = np.exp(logits)
		probs = odds / np.sum(odds) ###put this just before the for?

		return ((value, probs, allowedActions))


	def evaluateLeaf(self, leaf, value, done, breadcrumbs):

		lg.logger_mcts.debug('------EVALUATING LEAF------')

		if not done:
	
			state = GameState.from_id(leaf.state_id, config.GRID_SHAPE)
			value, probs, allowedActions = self.get_preds(state)
			lg.logger_mcts.debug('PREDICTED VALUE FOR %d: %f', state.currentPlayer, value)

			for idx, allowedAction in enumerate(allowedActions):
				if allowedAction:
					newState, _, _ = state.takeAction(idx)
					if newState.id not in self.mcts.tree:
						node = mc.Node(newState)
						self.mcts.addNode(node)
						lg.logger_mcts.debug('added node...%s...p = %f', node.state_id, probs[idx])
					else:
						node = self.mcts.tree[newState.id]
						lg.logger_mcts.debug('existing node...%s...', node.state_id)

					newEdge = mc.Edge(leaf, node, probs[idx], idx)
					leaf.edges.append((idx, newEdge))
				
		else:
			lg.logger_mcts.debug('GAME VALUE FOR %d: %f', GameState.current_player_from_id(leaf.state_id), value)

		return ((value, breadcrumbs))


		
	def getAV(self, tau):
		edges = self.mcts.root.edges
		pi = np.zeros(self.action_size, dtype=np.integer)
		values = np.zeros(self.action_size, dtype=np.float32)
		
		for action, edge in edges:
			pi[action] = pow(edge.stats['N'], 1/tau)
			values[action] = edge.stats['Q']

		pi = pi / (np.sum(pi) * 1.0)
		return pi, values

	def chooseAction(self, pi, values, tau):
		if tau == 0:
			actions = np.argwhere(pi == max(pi))
			action = random.choice(actions)[0]
		else:
			action_idx = np.random.multinomial(1, pi)
			action = np.where(action_idx==1)[0][0]

		value = values[action]

		return action, value

	def replay(self, ltmemory):
		lg.logger_mcts.debug('******RETRAINING MODEL******')


		for i in range(config.TRAINING_LOOPS):
			training_states = np.array([[row['state'].get_board_for_neural_network()] for row in ltmemory])
			training_targets = {VALUE_HEAD: np.array([row['value'] for row in ltmemory])
								, POLICY_HEAD: np.array([row['AV'] for row in ltmemory])} 

			fit = self.model.fit(training_states, training_targets, epochs=config.EPOCHS, verbose=1, validation_split=0, batch_size = config.BATCH_SIZE)
			lg.logger_mcts.debug('NEW LOSS %s', fit.history)

			self.train_overall_loss.append(round(fit.history['loss'][config.EPOCHS - 1],4))
			self.train_value_loss.append(round(fit.history['value_head_loss'][config.EPOCHS - 1],4)) 
			self.train_policy_loss.append(round(fit.history['policy_head_loss'][config.EPOCHS - 1],4)) 


	def buildMCTS(self, state):
		lg.logger_mcts.debug('****** BUILDING NEW MCTS TREE FOR AGENT %s ******', self.name)
		self.root = mc.Node(state)
		self.mcts = mc.MCTS(self.root, self.cpuct)

	def changeRootMCTS(self, state):
		lg.logger_mcts.debug('****** CHANGING ROOT OF MCTS TREE TO %s FOR AGENT %s ******', state.id, self.name)
		self.mcts.root = self.mcts.tree[state.id]