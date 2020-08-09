#### SELF PLAY
EPISODES = 30
MCTS_SIMS = 50
MEMORY_SIZE = 30000
TURNS_UNTIL_TAU0 = 10 # turn on which it starts playing deterministically
# c_puct: a number in (0, inf) that controls how quickly exploration
#            converges to the maximum-value policy. A higher value means
#            relying on the prior more.
CPUCT = 1
EPSILON = 0.2
ALPHA = 0.8


#### RETRAINING
BATCH_SIZE = 256
EPOCHS = 1
# Regularization constant
REG_CONST = 0.0001
LEARNING_RATE = 0.1
MOMENTUM = 0.9
TRAINING_LOOPS = 10

HIDDEN_CNN_LAYERS = [
	{'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	 , {'filters':75, 'kernel_size': (4,4)}
	]

#### EVALUATION
EVAL_EPISODES = 20
SCORING_THRESHOLD = 1.3

#### Settings
run_folder = './run/'
run_archive_folder = './run_archive/'

#### Game config 

PLAYER_1	= 1
NONE		= 0
PLAYER_2	= -1
RENDER_PLAYERS = {PLAYER_1:'X', NONE: '-', PLAYER_2:'O'}

NB_TOKENS_VICTORY = 4

# Purely cosmetic: points per victory, points per defeat
POINTS_VICTORY = 1
POINTS_DEFEAT = -1

# 0 and 1 are the "official" values, probably due to the way the neural network is working
VALUE_DEFAULT_ACTION = 0 # -1
VALUE_VICTORY = 1  # 100

GRID_SHAPE = (6,7)