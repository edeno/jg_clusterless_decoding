
from logging import getLogger

import numpy as np
import pandas as pd
from loren_frank_data_processing.position import _get_pos_dataframe
from src.parameters import ANIMALS, EDGE_ORDER, EDGE_SPACING
from track_linearization import get_linearized_position, make_track_graph

logger = getLogger(__name__)


def get_track_graph():
    NODE_POSITIONS = np.asarray([
        (55.2, 167),
        (91.3, 29),
        (29.2, 30.9),
        (60.2, 31.3),
        (24.8, 95.1),
        (91.7, 94.2),
        (56.8, 98.3)
    ])

    EDGES = np.asarray([
        (3, 6),
        (0, 6),
        (6, 4),
        (4, 2),
        (6, 5),
        (5, 1)
    ])

    return make_track_graph(NODE_POSITIONS, EDGES)


def get_interpolated_position_info(
        epoch_key, animals, use_HMM=False,
        route_euclidean_distance_scaling=1.0, sensor_std_dev=5.0,
        diagonal_bias=0.5, edge_spacing=EDGE_SPACING,
        edge_order=EDGE_ORDER):

    position_info = _get_pos_dataframe(epoch_key, animals)
    position_info = position_info.resample('2ms').mean().interpolate('time')
    position_info.loc[
        position_info.speed < 0, 'speed'] = 0.0
    track_graph = get_track_graph()
    position = np.asarray(position_info.loc[:, ['x_position', 'y_position']])

    linear_position_df = get_linearized_position(
        position=position,
        track_graph=track_graph,
        edge_spacing=EDGE_SPACING,
        edge_order=EDGE_ORDER,
        use_HMM=use_HMM,
        route_euclidean_distance_scaling=route_euclidean_distance_scaling,
        sensor_std_dev=sensor_std_dev,
        diagonal_bias=diagonal_bias,
    ).set_index(position_info.index)

    position_info = pd.concat((position_info, linear_position_df), axis=1)

    return position_info


def load_data(epoch_key):
    logger.info('Loading position information and linearizing...')
    position_info = (get_interpolated_position_info(epoch_key, ANIMALS)
                     .dropna(subset=["linear_position"]))

    return {
        'position_info': position_info,
    }
