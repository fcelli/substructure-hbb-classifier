import numpy as np


hyper_pars = {
    'penalty' : ['l1', 'l2'],
    'C' : np.logspace(-4, 4, 100),
    'solver' : ['liblinear']
}
