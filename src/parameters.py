from os.path import abspath, dirname, join, pardir

import numpy as np
from loren_frank_data_processing import Animal

# LFP sampling frequency
SAMPLING_FREQUENCY = 500

# Data directories and definitions
ROOT_DIR = join(abspath(dirname(__file__)), pardir)
RAW_DATA_DIR = join(ROOT_DIR, 'Raw-Data')
PROCESSED_DATA_DIR = join(ROOT_DIR, 'Processed-Data')
FIGURE_DIR = join(ROOT_DIR, 'figures')

ANIMALS = {
    'peanut': Animal(directory='/stelmo/jguidera/peanut/filterframework',
                     short_name='peanut'),
}

EDGE_ORDER = np.asarray([
    (3, 6),
    (0, 6),
    (6, 4),
    (4, 2),
    (6, 5),
    (5, 1)
])

EDGE_SPACING = 15
