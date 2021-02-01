import logging
import subprocess
import sys
from argparse import ArgumentParser

import numpy as np
import xarray as xr
from dask.distributed import Client
from replay_trajectory_classification import SortedSpikesClassifier
from sklearn.model_selection import KFold
from src.load_data import load_data
from src.parameters import (EDGE_ORDER, EDGE_SPACING, classifier_parameters,
                            state_names)

logging.basicConfig(level="INFO",
                    format="%(asctime)s %(message)s",
                    datefmt="%d-%b-%y %H:%M:%S")


def sorted_spikes_1D_decoding(epoch_key):
    data = load_data(epoch_key)

    logging.info(data["neuron_info"].area.value_counts())

    for area in data["neuron_info"].area.unique():
        logging.info(f"Decoding {area}...")

        cv = KFold()
        results = []

        neuron_id = (data["neuron_info"]
                     .loc[data["neuron_info"].area == area]
                     .neuron_id)
        spikes = data["spikes"].loc[:, neuron_id]

        for fold_ind, (train, test) in enumerate(
                cv.split(data["position_info"].index)):
            logging.info(f"Fitting Fold #{fold_ind + 1}...")
            classifier = SortedSpikesClassifier(**classifier_parameters)
            classifier.fit(
                position=data["position_info"].iloc[train].linear_position,
                spikes=spikes.iloc[train],
                track_graph=data["track_graph"],
                edge_order=EDGE_ORDER,
                edge_spacing=EDGE_SPACING,
            )

            logging.info("Predicting posterior...")
            results.append(
                classifier.predict(
                    spikes.iloc[test],
                    time=data["position_info"].iloc[test].index /
                    np.timedelta64(1, "s"),
                    state_names=state_names,
                )
            )
            classifier.save_model(
                f"{epoch_key[0]}_{epoch_key[1]:02d}_{epoch_key[2]:02d}_"
                f"sortedspikes_{area}_model_fold{fold_ind}.pkl"
            )

        # concatenate cv classifier results
        results = xr.concat(results, dim="time")

        results.to_netcdf(
            f"{epoch_key[0]}_{epoch_key[1]:02d}_{epoch_key[2]:02d}_"
            f"sortedspikes_{area}_results.nc"
        )

    logging.info("Done...\n\n")


def clusterless_1D_decoding(data):
    pass


def get_command_line_arguments():
    parser = ArgumentParser()
    parser.add_argument('Animal', type=str, help='Short name of animal')
    parser.add_argument('Day', type=int, help='Day of recording session')
    parser.add_argument('Epoch', type=int,
                        help='Epoch number of recording session')
    parser.add_argument('--data_type', type=str, default='sorted_spikes')
    return parser.parse_args()


analysis_types = {
    'sorted_spikes': sorted_spikes_1D_decoding,
    'clusterless': clusterless_1D_decoding,
}


def main():
    args = get_command_line_arguments()

    epoch_key = (args.Animal, args.Day, args.Epoch)
    logging.info(
        f'Processing epoch: Animal {args.Animal}, Day {args.Day},'
        f'Epoch #{args.Epoch}...')
    logging.info(f'Data type: {args.data_type}')
    git_hash = subprocess.run(['git', 'rev-parse', 'HEAD'],
                              stdout=subprocess.PIPE,
                              universal_newlines=True).stdout
    logging.info(f'Git Hash: {git_hash.rstrip()}')

    # Analysis Code
    client_params = dict(n_workers=48, threads_per_worker=2, processes=True)
    with Client(**client_params) as client:
        logging.info(client)
        analysis_types[args.data_type](epoch_key)


if __name__ == '__main__':
    sys.exit(main())
