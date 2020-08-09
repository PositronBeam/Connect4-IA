import unittest
import numpy as np

from game import Game, GameState
from config import PLAYER_1, PLAYER_2, NONE, NB_TOKENS_VICTORY, VALUE_DEFAULT_ACTION, VALUE_VICTORY, GRID_SHAPE

class TestGameMethods(unittest.TestCase):

    def test_init_render(self):
        env = Game()
        init_render = env.gameState.render()
        init_render_theory = "['-', '-', '-', '-', '-', '-', '-']\n" \
                            "['-', '-', '-', '-', '-', '-', '-']\n" \
                            "['-', '-', '-', '-', '-', '-', '-']\n" \
                            "['-', '-', '-', '-', '-', '-', '-']\n" \
                            "['-', '-', '-', '-', '-', '-', '-']\n" \
                            "['-', '-', '-', '-', '-', '-', '-']\n"
        self.assertEqual(init_render, init_render_theory)

    def test_action_after_init(self):
        env = Game()
        next_state, value, done = env.step(1)
        next_state_render = next_state.render()
        next_render_theory = "['-', '-', '-', '-', '-', '-', '-']\n" \
                            "['-', '-', '-', '-', '-', '-', '-']\n" \
                            "['-', '-', '-', '-', '-', '-', '-']\n" \
                            "['-', '-', '-', '-', '-', '-', '-']\n" \
                            "['-', '-', '-', '-', '-', '-', '-']\n" \
                            "['-', 'X', '-', '-', '-', '-', '-']\n"
        self.assertEqual(next_state_render, next_render_theory)

    def test_2_actions_after_init(self):
        env = Game()
        env.step(1)
        next_state, value, done = env.step(1)
        next_state_render = next_state.render()
        next_render_theory = "['-', '-', '-', '-', '-', '-', '-']\n" \
                            "['-', '-', '-', '-', '-', '-', '-']\n" \
                            "['-', '-', '-', '-', '-', '-', '-']\n" \
                            "['-', '-', '-', '-', '-', '-', '-']\n" \
                            "['-', 'O', '-', '-', '-', '-', '-']\n" \
                            "['-', 'X', '-', '-', '-', '-', '-']\n"
        self.assertEqual(next_state_render, next_render_theory)

    
    def test_end_game(self):

        grid_shape = GRID_SHAPE
        game_state = GameState(grid_shape = grid_shape, currentPlayer=PLAYER_1, board=np.full(grid_shape, PLAYER_1, dtype=np.int8))
        is_end_game = game_state.checkForEndGame()
        self.assertEqual(is_end_game, True)

        game_state = GameState(grid_shape = grid_shape, currentPlayer=PLAYER_1, board=np.full(grid_shape, NONE, dtype=np.int8))
        is_end_game = game_state.checkForEndGame()
        self.assertEqual(is_end_game, False)


    def test_victory_false_1(self):

        grid_shape = GRID_SHAPE
        board = np.full(grid_shape, NONE, dtype=np.int8)
        board[0, 0] = PLAYER_1
        game_state = GameState(grid_shape = grid_shape, currentPlayer=PLAYER_1, board=board)
        is_victory = game_state._isVictory(0, PLAYER_1)
        self.assertEqual(is_victory, False)

    def test_victory_false_2(self):

        grid_shape = GRID_SHAPE
        board = np.full(grid_shape, NONE, dtype=np.int8)
        board[0, 6] = PLAYER_1
        game_state = GameState(grid_shape = grid_shape, currentPlayer=PLAYER_1, board=board)
        is_victory = game_state._isVictory(6, PLAYER_1)
        self.assertEqual(is_victory, False)

    def test_victory_false_3(self):

        grid_shape = GRID_SHAPE
        board = np.full(grid_shape, NONE, dtype=np.int8)
        board[0, 3] = PLAYER_1
        game_state = GameState(grid_shape = grid_shape, currentPlayer=PLAYER_1, board=board)
        is_victory = game_state._isVictory(3, PLAYER_1)
        self.assertEqual(is_victory, False)

    def test_victory_true_1(self):
        """Victoires en ligne"""

        grid_shape = GRID_SHAPE
        for ligne in range(0,6):
            #On met toutes les façons d'aligner 4 PLAYER_1 dans la ligne ligne
            for column in range(0,GRID_SHAPE[1]+1-NB_TOKENS_VICTORY):
                board = np.full(grid_shape, NONE, dtype=np.int8)
                board[ligne, column:column+NB_TOKENS_VICTORY] = PLAYER_1
                #On remplit les lignes sous ligne avec PLAYER_2
                board[0:ligne,:] = PLAYER_2

                game_state = GameState(grid_shape = grid_shape, currentPlayer=PLAYER_1, board=board)

                for i in range(column,column+NB_TOKENS_VICTORY):
                    is_victory = game_state._isVictory(i, PLAYER_1)
                    self.assertEqual(is_victory, True)



    def test_victory_true_2(self):
        """Victoires en colonne"""

        grid_shape = GRID_SHAPE

        for ligne in range(0,GRID_SHAPE[0]+1-NB_TOKENS_VICTORY):
            for column in range(0,GRID_SHAPE[1]):
                #On met toutes les façons d'aligner 4 PLAYER_1 dans la colonne column
                board = np.full(grid_shape, NONE, dtype=np.int8)
                board[ligne:ligne+NB_TOKENS_VICTORY, column] = PLAYER_1
                #On remplit les lignes sous ligne avec PLAYER_2
                board[0:ligne,:] = PLAYER_2
                game_state = GameState(grid_shape = grid_shape, currentPlayer=PLAYER_1, board=board)

                is_victory = game_state._isVictory(column, PLAYER_1)
                self.assertEqual(is_victory, True)

    def test_victory_true_3(self):
        """Victoires en diagonale 1 (bas gauche, haut droit)"""

        grid_shape = GRID_SHAPE

        for ligne in range(0,GRID_SHAPE[0]+1-NB_TOKENS_VICTORY):
            for column in range(0,GRID_SHAPE[1]+1-NB_TOKENS_VICTORY):
                #On aligne 4 PLAYER_1 dans la diagonale dont le point de départ (bas,gauche) est (ligne,column)
                board = np.full(grid_shape, NONE, dtype=np.int8)
                for i,j in zip(range(ligne,ligne+NB_TOKENS_VICTORY), range(column, column+NB_TOKENS_VICTORY)):
                    board[i,j] = PLAYER_1
                    board[0:i,j] = PLAYER_2
                    board[i,j+1:] = PLAYER_2
                #On remplit les lignes sous ligne avec PLAYER_2
                board[0:ligne,:] = PLAYER_2
                game_state = GameState(grid_shape = grid_shape, currentPlayer=PLAYER_1, board=board)

                for i in range(column,column+NB_TOKENS_VICTORY):
                    is_victory = game_state._isVictory(i, PLAYER_1)
                    self.assertEqual(is_victory, True)

    def test_victory_true_4(self):
        """Victoires en diagonale 2 (haut gauche, bas droit)"""

        grid_shape = GRID_SHAPE

        for ligne in range(GRID_SHAPE[0]+1-NB_TOKENS_VICTORY,GRID_SHAPE[0]):
            for column in range(0,GRID_SHAPE[1]+1-NB_TOKENS_VICTORY):
                #On aligne 4 PLAYER_1 dans la diagonale dont le point de départ (bas,gauche) est (ligne,column)
                board = np.full(grid_shape, NONE, dtype=np.int8)
                for i,j in zip(range(ligne,ligne-NB_TOKENS_VICTORY,-1), range(column, column+NB_TOKENS_VICTORY)):
                    board[i,j] = PLAYER_1
                    board[0:i,j] = PLAYER_2
                    board[i,0:j] = PLAYER_2
                #On remplit les lignes sous ligne avec PLAYER_2
                board[0:ligne-3,:] = PLAYER_2
                game_state = GameState(grid_shape = grid_shape, currentPlayer=PLAYER_1, board=board)

                for i in range(column,column+NB_TOKENS_VICTORY):
                    is_victory = game_state._isVictory(i, PLAYER_1)
                    self.assertEqual(is_victory, True)

    def test_id(self):
        """Je transforme les victoires de test_victory_true_4 en id"""
        """Puis je crée un GameState à partir de cet id"""
        """Et je teste si le nouveau GameState est égal à l'ancien"""

        for ligne in range(GRID_SHAPE[0]+1-NB_TOKENS_VICTORY,GRID_SHAPE[0]):
            for column in range(0,GRID_SHAPE[1]+1-NB_TOKENS_VICTORY):
                #On aligne 4 PLAYER_1 dans la diagonale dont le point de départ (bas,gauche) est (ligne,column)
                board = np.full(GRID_SHAPE, NONE, dtype=np.int8)
                for i,j in zip(range(ligne,ligne-NB_TOKENS_VICTORY,-1), range(column, column+NB_TOKENS_VICTORY)):
                    board[i,j] = PLAYER_1
                    board[0:i,j] = PLAYER_2
                    board[i,0:j] = PLAYER_2
                #On remplit les lignes sous ligne avec PLAYER_2
                board[0:ligne-3,:] = PLAYER_2
                game_state = GameState(currentPlayer=PLAYER_1, board=board)

                id = game_state.id

                new_game_state = GameState.from_id(id, board.shape)
                
                # See https://stackoverflow.com/questions/3302949/best-way-to-assert-for-numpy-array-equality
                self.assertIsNone(np.testing.assert_array_equal(new_game_state.board, game_state.board))
                self.assertEqual(new_game_state.currentPlayer, game_state.currentPlayer)

        
    def test_game_victory(self):

        env = Game()

        env.step(1)
        next_state, value, done = env.step(1)
        next_state_render = next_state.render()
        next_render_theory = "['-', '-', '-', '-', '-', '-', '-']\n" \
                            "['-', '-', '-', '-', '-', '-', '-']\n" \
                            "['-', '-', '-', '-', '-', '-', '-']\n" \
                            "['-', '-', '-', '-', '-', '-', '-']\n" \
                            "['-', 'O', '-', '-', '-', '-', '-']\n" \
                            "['-', 'X', '-', '-', '-', '-', '-']\n"
        self.assertEqual(next_state_render, next_render_theory)
        self.assertEqual(value, -VALUE_DEFAULT_ACTION)
        self.assertEqual(done, False)

        next_state, value, done = env.step(1)
        next_state_render = next_state.render()
        next_render_theory = "['-', '-', '-', '-', '-', '-', '-']\n" \
                            "['-', '-', '-', '-', '-', '-', '-']\n" \
                            "['-', '-', '-', '-', '-', '-', '-']\n" \
                            "['-', 'X', '-', '-', '-', '-', '-']\n" \
                            "['-', 'O', '-', '-', '-', '-', '-']\n" \
                            "['-', 'X', '-', '-', '-', '-', '-']\n"
        self.assertEqual(next_state_render, next_render_theory)
        self.assertEqual(value, -VALUE_DEFAULT_ACTION)
        self.assertEqual(done, False)

        next_state, value, done = env.step(1)
        next_state_render = next_state.render()
        next_render_theory = "['-', '-', '-', '-', '-', '-', '-']\n" \
                            "['-', '-', '-', '-', '-', '-', '-']\n" \
                            "['-', 'O', '-', '-', '-', '-', '-']\n" \
                            "['-', 'X', '-', '-', '-', '-', '-']\n" \
                            "['-', 'O', '-', '-', '-', '-', '-']\n" \
                            "['-', 'X', '-', '-', '-', '-', '-']\n"
        self.assertEqual(next_state_render, next_render_theory)
        self.assertEqual(value, -VALUE_DEFAULT_ACTION)
        self.assertEqual(done, False)

        next_state, value, done = env.step(1)
        next_state_render = next_state.render()
        next_render_theory = "['-', '-', '-', '-', '-', '-', '-']\n" \
                            "['-', 'X', '-', '-', '-', '-', '-']\n" \
                            "['-', 'O', '-', '-', '-', '-', '-']\n" \
                            "['-', 'X', '-', '-', '-', '-', '-']\n" \
                            "['-', 'O', '-', '-', '-', '-', '-']\n" \
                            "['-', 'X', '-', '-', '-', '-', '-']\n"
        self.assertEqual(next_state_render, next_render_theory)
        self.assertEqual(value, -VALUE_DEFAULT_ACTION)
        self.assertEqual(done, False)

        next_state, value, done = env.step(1)
        next_state_render = next_state.render()
        next_render_theory = "['-', 'O', '-', '-', '-', '-', '-']\n" \
                            "['-', 'X', '-', '-', '-', '-', '-']\n" \
                            "['-', 'O', '-', '-', '-', '-', '-']\n" \
                            "['-', 'X', '-', '-', '-', '-', '-']\n" \
                            "['-', 'O', '-', '-', '-', '-', '-']\n" \
                            "['-', 'X', '-', '-', '-', '-', '-']\n"
        self.assertEqual(next_state_render, next_render_theory)
        self.assertEqual(value, -VALUE_DEFAULT_ACTION)
        self.assertEqual(done, False)

        next_state, value, done = env.step(2)
        next_state_render = next_state.render()
        next_render_theory = "['-', 'O', '-', '-', '-', '-', '-']\n" \
                            "['-', 'X', '-', '-', '-', '-', '-']\n" \
                            "['-', 'O', '-', '-', '-', '-', '-']\n" \
                            "['-', 'X', '-', '-', '-', '-', '-']\n" \
                            "['-', 'O', '-', '-', '-', '-', '-']\n" \
                            "['-', 'X', 'X', '-', '-', '-', '-']\n"
        self.assertEqual(next_state_render, next_render_theory)
        self.assertEqual(value, -VALUE_DEFAULT_ACTION)
        self.assertEqual(done, False)

        next_state, value, done = env.step(3)
        next_state_render = next_state.render()
        next_render_theory = "['-', 'O', '-', '-', '-', '-', '-']\n" \
                            "['-', 'X', '-', '-', '-', '-', '-']\n" \
                            "['-', 'O', '-', '-', '-', '-', '-']\n" \
                            "['-', 'X', '-', '-', '-', '-', '-']\n" \
                            "['-', 'O', '-', '-', '-', '-', '-']\n" \
                            "['-', 'X', 'X', 'O', '-', '-', '-']\n"
        self.assertEqual(next_state_render, next_render_theory)
        self.assertEqual(value, -VALUE_DEFAULT_ACTION)
        self.assertEqual(done, False)


        next_state, value, done = env.step(3)
        next_state_render = next_state.render()
        next_render_theory = "['-', 'O', '-', '-', '-', '-', '-']\n" \
                            "['-', 'X', '-', '-', '-', '-', '-']\n" \
                            "['-', 'O', '-', '-', '-', '-', '-']\n" \
                            "['-', 'X', '-', '-', '-', '-', '-']\n" \
                            "['-', 'O', '-', 'X', '-', '-', '-']\n" \
                            "['-', 'X', 'X', 'O', '-', '-', '-']\n"
        self.assertEqual(next_state_render, next_render_theory)
        self.assertEqual(value, -VALUE_DEFAULT_ACTION)
        self.assertEqual(done, False)

        next_state, value, done = env.step(2)
        next_state_render = next_state.render()
        next_render_theory = "['-', 'O', '-', '-', '-', '-', '-']\n" \
                            "['-', 'X', '-', '-', '-', '-', '-']\n" \
                            "['-', 'O', '-', '-', '-', '-', '-']\n" \
                            "['-', 'X', '-', '-', '-', '-', '-']\n" \
                            "['-', 'O', 'O', 'X', '-', '-', '-']\n" \
                            "['-', 'X', 'X', 'O', '-', '-', '-']\n"
        self.assertEqual(next_state_render, next_render_theory)
        self.assertEqual(value, -VALUE_DEFAULT_ACTION)
        self.assertEqual(done, False)

        next_state, value, done = env.step(2)
        next_state_render = next_state.render()
        next_render_theory = "['-', 'O', '-', '-', '-', '-', '-']\n" \
                            "['-', 'X', '-', '-', '-', '-', '-']\n" \
                            "['-', 'O', '-', '-', '-', '-', '-']\n" \
                            "['-', 'X', 'X', '-', '-', '-', '-']\n" \
                            "['-', 'O', 'O', 'X', '-', '-', '-']\n" \
                            "['-', 'X', 'X', 'O', '-', '-', '-']\n"
        self.assertEqual(next_state_render, next_render_theory)
        self.assertEqual(value, -VALUE_DEFAULT_ACTION)
        self.assertEqual(done, False)

        next_state, value, done = env.step(2)
        next_state_render = next_state.render()
        next_render_theory = "['-', 'O', '-', '-', '-', '-', '-']\n" \
                            "['-', 'X', '-', '-', '-', '-', '-']\n" \
                            "['-', 'O', 'O', '-', '-', '-', '-']\n" \
                            "['-', 'X', 'X', '-', '-', '-', '-']\n" \
                            "['-', 'O', 'O', 'X', '-', '-', '-']\n" \
                            "['-', 'X', 'X', 'O', '-', '-', '-']\n"
        self.assertEqual(next_state_render, next_render_theory)
        self.assertEqual(value, -VALUE_DEFAULT_ACTION)
        self.assertEqual(done, False)

        next_state, value, done = env.step(3)
        next_state_render = next_state.render()
        next_render_theory = "['-', 'O', '-', '-', '-', '-', '-']\n" \
                            "['-', 'X', '-', '-', '-', '-', '-']\n" \
                            "['-', 'O', 'O', '-', '-', '-', '-']\n" \
                            "['-', 'X', 'X', 'X', '-', '-', '-']\n" \
                            "['-', 'O', 'O', 'X', '-', '-', '-']\n" \
                            "['-', 'X', 'X', 'O', '-', '-', '-']\n"
        self.assertEqual(next_state_render, next_render_theory)
        self.assertEqual(value, -VALUE_DEFAULT_ACTION)
        self.assertEqual(done, False)

        next_state, value, done = env.step(4)
        next_state_render = next_state.render()
        next_render_theory = "['-', 'O', '-', '-', '-', '-', '-']\n" \
                            "['-', 'X', '-', '-', '-', '-', '-']\n" \
                            "['-', 'O', 'O', '-', '-', '-', '-']\n" \
                            "['-', 'X', 'X', 'X', '-', '-', '-']\n" \
                            "['-', 'O', 'O', 'X', '-', '-', '-']\n" \
                            "['-', 'X', 'X', 'O', 'O', '-', '-']\n"
        self.assertEqual(next_state_render, next_render_theory)
        self.assertEqual(value, -VALUE_DEFAULT_ACTION)
        self.assertEqual(done, False)

        next_state, value, done = env.step(4)
        next_state_render = next_state.render()
        next_render_theory = "['-', 'O', '-', '-', '-', '-', '-']\n" \
                            "['-', 'X', '-', '-', '-', '-', '-']\n" \
                            "['-', 'O', 'O', '-', '-', '-', '-']\n" \
                            "['-', 'X', 'X', 'X', '-', '-', '-']\n" \
                            "['-', 'O', 'O', 'X', 'X', '-', '-']\n" \
                            "['-', 'X', 'X', 'O', 'O', '-', '-']\n"
        self.assertEqual(next_state_render, next_render_theory)
        self.assertEqual(value, -VALUE_DEFAULT_ACTION)
        self.assertEqual(done, False)

        next_state, value, done = env.step(5)
        next_state_render = next_state.render()
        next_render_theory = "['-', 'O', '-', '-', '-', '-', '-']\n" \
                            "['-', 'X', '-', '-', '-', '-', '-']\n" \
                            "['-', 'O', 'O', '-', '-', '-', '-']\n" \
                            "['-', 'X', 'X', 'X', '-', '-', '-']\n" \
                            "['-', 'O', 'O', 'X', 'X', '-', '-']\n" \
                            "['-', 'X', 'X', 'O', 'O', 'O', '-']\n"
        self.assertEqual(next_state_render, next_render_theory)
        self.assertEqual(value, -VALUE_DEFAULT_ACTION)
        self.assertEqual(done, False)

        next_state, value, done = env.step(4)
        next_state_render = next_state.render()
        next_render_theory = "['-', 'O', '-', '-', '-', '-', '-']\n" \
                            "['-', 'X', '-', '-', '-', '-', '-']\n" \
                            "['-', 'O', 'O', '-', '-', '-', '-']\n" \
                            "['-', 'X', 'X', 'X', 'X', '-', '-']\n" \
                            "['-', 'O', 'O', 'X', 'X', '-', '-']\n" \
                            "['-', 'X', 'X', 'O', 'O', 'O', '-']\n"
        self.assertEqual(next_state_render, next_render_theory)
        self.assertEqual(value, -VALUE_VICTORY)
        self.assertEqual(done, True)


if __name__ == '__main__':
    unittest.main()