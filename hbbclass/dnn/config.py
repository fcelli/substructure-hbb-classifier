# Directories
OUTPUT_PATH = './data/models/dnn/'
TUNING_NAME = 'dnn_tuning'
MODEL_NAME  = 'dnn_model'
# Model
INPUT_SHAPE = (4,)
SEED = 42
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 5
# Hyperparameters
BATCH_SIZE = [16, 32, 64]
LEARNING_RATES = [1e-2, 1e-3, 1e-4, 1e-5]
N_LAYERS_LOW = 1
N_LAYERS_HIGH = 5
N_UNITS_LOW = 8
N_UNITS_HIGH = 160
N_UNITS_STEP = 8
