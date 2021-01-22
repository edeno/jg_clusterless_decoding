
from logging import getLogger

import numpy as np
import pandas as pd
from loren_frank_data_processing import (get_all_multiunit_indicators,
                                         make_tetrode_dataframe)
from loren_frank_data_processing.core import get_data_structure
from ripple_detection import get_multiunit_population_firing_rate
from src.parameters import (ANIMALS, EDGE_ORDER, EDGE_SPACING,
                            SAMPLING_FREQUENCY)
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


def _get_pos_dataframe(epoch_key, animals):
    animal, day, epoch = epoch_key
    struct = get_data_structure(animals[animal], day, 'pos', 'pos')[epoch - 1]
    position_data = struct['data'][0, 0]
    FIELD_NAMES = ['time', 'x_position', 'y_position', 'head_direction',
                   'speed', 'smoothed_x_position', 'smoothed_y_position',
                   'smoothed_head_direction', 'smoothed_speed']
    time = pd.TimedeltaIndex(
        position_data[:, 0], unit='s', name='time')
    n_cols = position_data.shape[1]

    if n_cols > 5:
        # Use the smoothed data if available
        NEW_NAMES = {'smoothed_x_position': 'x_position',
                     'smoothed_y_position': 'y_position',
                     'smoothed_head_direction': 'head_direction',
                     'smoothed_speed': 'speed'}
        return (pd.DataFrame(
            position_data[:, 5:9], columns=FIELD_NAMES[5:9], index=time)
            .rename(columns=NEW_NAMES))
    else:
        return pd.DataFrame(position_data[:, 1:5], columns=FIELD_NAMES[1:5],
                            index=time)


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
    track_graph = get_track_graph()

    logger.info('Loading multiunits...')
    tetrode_info = make_tetrode_dataframe(
        ANIMALS).xs(epoch_key, drop_level=False)
    
    def n_dead_chans(x):
        if isinstance(x, float):
            return 1
        elif isinstance(x, (list, tuple, np.ndarray)):
            return len(x)


    bad_trode = [9, 16, 21]

    tetrode_keys = tetrode_info.loc[
        (tetrode_info.area == "CA1")
        & (tetrode_info.deadchans.apply(lambda x: n_dead_chans(x)) < 4)
        & ~tetrode_info.nTrode.isin(tetrode_info.ref.dropna().unique())
        & ~tetrode_info.nTrode.isin(bad_trode)
    ].index

    def _time_function(*args, **kwargs):
        return position_info.index

    multiunits = get_all_multiunit_indicators(
        tetrode_keys, ANIMALS, _time_function)

    multiunit_spikes = (np.any(~np.isnan(multiunits.values), axis=1)
                        ).astype(np.float)
    multiunit_firing_rate = pd.DataFrame(
        get_multiunit_population_firing_rate(
            multiunit_spikes, SAMPLING_FREQUENCY), index=position_info.index,
        columns=['firing_rate'])

    return {
        'position_info': position_info,
        'tetrode_info': tetrode_info,
        'multiunits': multiunits,
        'multiunit_firing_rate': multiunit_firing_rate,
        'track_graph': track_graph,
    }
