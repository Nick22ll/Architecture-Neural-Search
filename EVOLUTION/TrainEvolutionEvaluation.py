import math
import os
import pickle as pkl
import sys

import numpy as np
import torch
import torch.nn as nn

from CIFAR10Dataset import CIFAR10Dataset
from Cells import NormalCell
from EvaluateCellNetwork import evaluateCellNetwork
from PlotUtils import plot_training_statistics, plot_history
from time import time


def main():
    os.chdir(os.path.dirname(sys.argv[0]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = f"EVOLUTIONEvaluation"
    BEST_NETWORK = "Top0Net.pkl"
    with open(f"EVOLUTIONBestNetworks/{BEST_NETWORK}", "rb") as net_file:
        network = pkl.load(net_file)

    # normal_cell = network.operationList[0].edge_operations
    # reduction_cell = network.operationList[network.reduction_indices[0]].edge_operations
    #
    # del network
    #
    # EVALUATION_CELLS_NUM = 21
    # REDUCTION_POS = [math.floor(EVALUATION_CELLS_NUM / 3), math.floor(EVALUATION_CELLS_NUM / 3 * 2)]
    #
    # first_block = [normal_cell] * REDUCTION_POS[0]
    # second_block = [normal_cell] * REDUCTION_POS[0]
    # last_block = [normal_cell] * (EVALUATION_CELLS_NUM - (2 * REDUCTION_POS[0]))
    # new_cells_list = first_block + [reduction_cell] + second_block + [reduction_cell] + last_block

    new_cell_list = []
    for cell in network.operationList[:-1]:
        if isinstance(cell, NormalCell):
            new_cell_list.append(("normal", cell.edge_operations))
        else:
            new_cell_list.append(("reduction", cell.edge_operations))

    evaluateCellNetwork(new_cell_list, save_path, device)


if __name__ == "__main__":
    main()
