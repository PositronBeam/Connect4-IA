import numpy as np
from collections import deque

import config

class Memory:
	def __init__(self, MEMORY_SIZE):
		self.MEMORY_SIZE = config.MEMORY_SIZE
		# long term memory
		self.ltmemory = deque(maxlen=config.MEMORY_SIZE)
		# short term memory (commit_ltmemory() copies it to ltmemory and then deletes it)
		self.stmemory = deque(maxlen=config.MEMORY_SIZE)

	def commit_stmemory(self, state, actionValues):
		self.stmemory.append({ 'state': state, 'AV': actionValues })
		# Commit also the symetrical board and values
		symetrical_state = state.generate_symetric_state()
		symetrical_action_values = list(reversed(actionValues))
		self.stmemory.append({ 'state': symetrical_state, 'AV': symetrical_action_values })


	def commit_ltmemory(self):
		for i in self.stmemory:
			self.ltmemory.append(i)
		self.clear_stmemory()

	def clear_stmemory(self):
		self.stmemory = deque(maxlen=config.MEMORY_SIZE)

	def clear_ltmemory(self):
		self.ltmemory = deque(maxlen=config.MEMORY_SIZE)
		