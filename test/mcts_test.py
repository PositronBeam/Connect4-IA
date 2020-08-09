import unittest
import MCTS as mc

import config
from game import Game, GameState

class TestMCTSMethods(unittest.TestCase):

    def test_init_mcts(self):

        env = Game()
        root = mc.Node(env.gameState)
        mcts = mc.MCTS(root, config.CPUCT)
        self.assertEqual(mcts.root, root)
        self.assertEqual(mcts.tree[env.gameState._generate_id()], root)

    

