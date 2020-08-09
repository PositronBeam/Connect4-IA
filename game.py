import numpy as np
import functools

import config


class Game:

	def __init__(self, grid_shape = config.GRID_SHAPE):		
		self.grid_shape = grid_shape
		self.name = 'connect4'
		self.action_size = grid_shape[1]
		self.reset()

	def reset(self):
		self.gameState = GameState(grid_shape = self.grid_shape, currentPlayer = config.PLAYER_1)
		return self.gameState


	def step(self,action):
		next_state, value, done = self.gameState.takeAction(action)
		# updates current game state
		self.gameState = next_state

		return (next_state, value, done)




class GameState():

	def __init__(self, currentPlayer, grid_shape = None, board=None):

		self.currentPlayer = currentPlayer
		if board is not None:
			self.board = board
		else:
			self.board = np.full(grid_shape, config.NONE, dtype=np.int8)
		self.id = self._generate_id()

	def generate_symetric_state(self):

		#Astuce: pour prendre le symetrique de board par rapport à la colonne 3, on fait ceci:
		# on voudrait utiliser "reversed" pour inverser les colonnes, mais on ne peut pas le faire directement
		# car "reversed" inverse les lignes
		# => on transforme les colonnes en lignes en prenant la transpose, on inverse les lignes, et on retranspose
		symetric_board = np.array(list(reversed(self.board.T))).T
		
		symetric_game_state = GameState(self.currentPlayer, board=symetric_board)
		return symetric_game_state


	def get_board_for_neural_network(self):

		# TODO symétrie par rapport à l'axe central: on peut diviser par 2 les états possibles en
		# ne considérant que les états où il y a davantage de jetons à gauche de l'axe de symétrie. Pour ce faire:
		# quand on est dans un état où il y a davantage de jetons à droite, on calcule son symétrique, et c'est 
		# son symétrique qu'on renvoie. Par contre ensuite, il faut se souvenir que le réseau de neurones
		# voit le symétrique de l'état réel, car il faut prendre le symétrique de l'action qu'il va calculer

		# Si c'est au tour de PLAYER_2, on inverse le point de vue.
		# Ainsi, le réseau de neurones est toujours le joueur 1 et son adversaire est toujours le joueur -1
		result = self.board * self.currentPlayer

		return result

		

	def allowedActions(self):
		"""Returns [True, True, True, True, True, True, True]... one boolean per authorized action"""
		allowed_actions = (self.board[-1,:] == config.NONE)
		return allowed_actions

	def checkForEndGame(self):
		"""Returns true if no action is possible"""
		"""Does NOT check for victory of the players"""
		allowed_actions = self.allowedActions()
		result = all(p == False for p in allowed_actions)
		return result

	def takeAction(self, action):
		"""action must be between 0 (most left) and 6 (most right - ie. grid_shape[1]-1)"""
		"""Computes (newState, value, done)"""
		"""newState: GameState representing the state of the game after the current player has taken action"""
		"""value: reward for that action"""
		"""done: 1 if end of the game, 0 otherwise"""

		# See agent.evaluateLeaf(): the agent won't chose a forbidden action (ie. chose a full column), 
		# so we don't have to handle that case

		next_board = np.copy(self.board)
		# The bottom line has number 0, the top line has number 5 (ie. grid_shape[0]-1)
		column_content = next_board[:,action]
		indice_of_none = np.where(column_content == config.NONE)[0][0]
		column_content[indice_of_none] = self.currentPlayer

		next_state = GameState(currentPlayer=-self.currentPlayer, board=next_board)

		done = next_state._isVictory(action, self.currentPlayer)
		value = config.VALUE_VICTORY if done else config.VALUE_DEFAULT_ACTION

		if not done:
			done = next_state.checkForEndGame()

		return (next_state, -value, done) 

	def _generate_id(self):
		"""Computes a unique id for that state (another identical state will have the same id)"""
		"""The id is as small as possible for saving memory space (even so, it is a 85-bits integer)"""

		board_as_line = np.reshape(self.board, -1)
		player1_line = (board_as_line == config.PLAYER_1)
		player2_line = (board_as_line == config.PLAYER_2)
		result_line = np.concatenate((player1_line,player2_line,[self.currentPlayer==config.PLAYER_1])).tolist()
		# See https://stackoverflow.com/questions/25583312/changing-an-array-of-true-and-false-answers-to-a-hex-value-python
		result_line_val = functools.reduce(lambda byte, bit: byte*2 + bit, result_line, 0)
		return result_line_val

	@staticmethod
	def from_id(id, grid_shape):
		"""Generates a GameState from id"""

		# See https://stackoverflow.com/questions/33608280/convert-4-bit-integer-into-boolean-list/33608387
		nb_booleans = grid_shape[0] * grid_shape[1] * 2 + 1		
		full_state = np.flip(np.array([bool(id & (1<<n)) for n in range(nb_booleans)]))
		player1_board = full_state[:grid_shape[0] * grid_shape[1]].reshape(grid_shape[0], grid_shape[1])
		player2_board = full_state[grid_shape[0] * grid_shape[1]:-1].reshape(grid_shape[0], grid_shape[1])
		current_player = full_state[-1]

		board = player1_board * config.PLAYER_1 + player2_board * config.PLAYER_2
		current_player = config.PLAYER_1 if current_player else config.PLAYER_2

		result = GameState(current_player, board=board.astype(np.int8))
		return result

	@staticmethod
	def current_player_from_id(id):

		current_player = bool(id & (1<<0))
		current_player = config.PLAYER_1 if current_player else config.PLAYER_2
		return current_player


	def _isVictory(self, latestAction, currentPlayer):
		"""Returns True if latestAction led currentPlayer to victory"""

		# column range for the *start* of the loop
		min_column = max(0, latestAction-config.NB_TOKENS_VICTORY+1)
		max_column = min(self.board.shape[1]-config.NB_TOKENS_VICTORY, latestAction)
		
		column_content = self.board[:,latestAction]
		line_of_latest_action = None
		for i in range(self.board.shape[0]-1,-1,-1):
			if column_content[i] != config.NONE:
				line_of_latest_action = i
				break

		# Check victory in the line of latest action
		for column in range(min_column, max_column+1):
			tokens = self.board[line_of_latest_action,column:column+config.NB_TOKENS_VICTORY]
			victory = all(p == currentPlayer for p in tokens)
			if victory: 
				return True

		# Check victory in the column of latest action
		min_line = max(0, line_of_latest_action-config.NB_TOKENS_VICTORY+1)
		tokens = column_content[min_line:min_line+config.NB_TOKENS_VICTORY]
		victory = all(p == currentPlayer for p in tokens)
		if victory: 
			return True

		# Check victory in the first diagonal of latest action (bas gauche, haut droit)
		for line, column in zip(range(line_of_latest_action-config.NB_TOKENS_VICTORY+1,line_of_latest_action+1), range(latestAction-config.NB_TOKENS_VICTORY+1, latestAction+1)):
			if line <= self.board.shape[0]-config.NB_TOKENS_VICTORY and line >= 0 and column >=0 and column <= self.board.shape[1]-config.NB_TOKENS_VICTORY:
				tokens = [self.board[l,c] for l,c in zip(range(line,line+config.NB_TOKENS_VICTORY), range(column, column+config.NB_TOKENS_VICTORY))]
				victory = all(p == currentPlayer for p in tokens)
				if victory: 
					return True


		# Check victory in the second diagonal of latest action (haut gauche, bas droit)
		for line, column in zip(range(line_of_latest_action+config.NB_TOKENS_VICTORY-1,line_of_latest_action-1,-1), range(latestAction-config.NB_TOKENS_VICTORY+1, latestAction+1)):
			if line < self.board.shape[0] and line >= config.NB_TOKENS_VICTORY-1 and column >=0 and column <= self.board.shape[1]-config.NB_TOKENS_VICTORY:
				tokens = [self.board[l,c] for l,c in zip(range(line,line-config.NB_TOKENS_VICTORY,-1), range(column, column+config.NB_TOKENS_VICTORY))]
				victory = all(p == currentPlayer for p in tokens)
				if victory: 
					return True
		
		return False


	def render(self):

		result=''
		for r in reversed(range(self.board.shape[0])):
			result = result + str([config.RENDER_PLAYERS[x] for x in self.board[r]]) + '\n'

		return result