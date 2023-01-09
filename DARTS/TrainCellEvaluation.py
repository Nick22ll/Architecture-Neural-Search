import math
import os
import pickle as pkl
import sys
from time import time

import numpy as np
import torch
import torch.nn as nn

from CIFAR10Dataset import CIFAR10Dataset
from CellNetwork import CellNetwork
from DARTS.SearchCellNetwork import SearchCellNetwork

from DARTS.SearchNetwork import SearchNetwork
from EvaluateCellNetwork import evaluateCellNetwork
from PlotUtils import plot_training_statistics, plot_history


def main():
    os.chdir(os.path.dirname(sys.argv[0]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = f"DARTSResults/CellEvaluation"
    os.makedirs(save_path, exist_ok=True)

    CELLS_NUM = 7
    IMAGE_DIM = 32
    IN_CHANNELS = 3
    FIRST_CELL_CHANNELS = 16

    model = SearchCellNetwork(CELLS_NUM)
    model.initialization(IMAGE_DIM, IN_CHANNELS, FIRST_CELL_CHANNELS, device)
    model.load("DARTSResults/CellSearch/BestLossCell/network.pt")

    cells_list = model.genotypeCellNetworkDARTS()

    normal_cell = cells_list[0]
    reduction_cell = cells_list[model.reduction_indices[0]]

    del model

    EVALUATION_CELLS_NUM = 21
    REDUCTION_POS = [math.floor(EVALUATION_CELLS_NUM / 3), math.floor(EVALUATION_CELLS_NUM / 3 * 2)]

    first_block = [normal_cell] * REDUCTION_POS[0]
    second_block = [normal_cell] * REDUCTION_POS[0]
    last_block = [normal_cell] * (EVALUATION_CELLS_NUM - (2 * REDUCTION_POS[0]))
    new_cells_list = first_block + [reduction_cell] + second_block + [reduction_cell] + last_block

    evaluateCellNetwork(new_cells_list, save_path, device)


if __name__ == "__main__":
    main()
