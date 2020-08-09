
import config
import logging

def setup_logger(name, log_file, level=logging.INFO):

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)

    return logger

def setup_logger_console(name, log_file, level=logging.INFO):
    

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    # add ch to logger
    logger.addHandler(ch)

    return logger


### SET all LOGGER_DISABLED to True to disable logging
### WARNING: the mcts log file gets big quite quickly

LOGGER_DISABLED = {
'main':False
, 'memory':False
, 'tourney':False
, 'mcts':False
, 'model': False}


logger_mcts = setup_logger_console('logger_mcts', config.run_folder + 'logs/logger_mcts.log', level=logging.ERROR)
logger_mcts.disabled = LOGGER_DISABLED['mcts']

logger_main = setup_logger_console('logger_main', config.run_folder + 'logs/logger_main.log')
logger_main.disabled = LOGGER_DISABLED['main']

logger_tourney = setup_logger_console('logger_tourney', config.run_folder + 'logs/logger_tourney.log')
logger_tourney.disabled = LOGGER_DISABLED['tourney']

logger_memory = setup_logger_console('logger_memory', config.run_folder + 'logs/logger_memory.log')
logger_memory.disabled = LOGGER_DISABLED['memory']

logger_model = setup_logger_console('logger_model', config.run_folder + 'logs/logger_model.log')
logger_model.disabled = LOGGER_DISABLED['model']
 