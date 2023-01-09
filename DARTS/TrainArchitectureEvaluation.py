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

from DARTS.SearchNetwork import SearchNetwork
from EvaluateCellNetwork import evaluateCellNetwork
from PlotUtils import plot_training_statistics, plot_history


def main():
    os.chdir(os.path.dirname(sys.argv[0]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    CELLS_NUM = 7
    IMAGE_DIM = 32
    IN_CHANNELS = 3
    FIRST_CELL_CHANNELS = 16

    model = SearchNetwork(CELLS_NUM)
    model.initialization(IMAGE_DIM, IN_CHANNELS, FIRST_CELL_CHANNELS, device)
    model.load("DARTSResults/ArchitectureSearch/BestLossArchitecture/network.pt")

    cells_list = model.genotypeCellNetworkDARTS()
    del model
    network = CellNetwork()
    network.initialization(cells_list, 32, 3, 36)

    save_path = f"DARTSResults/ArchitectureEvaluationNOCAT"

    for idx, cell in enumerate(cells_list):
        if cell[0] == "reduction":
            network.operationList[idx].draw(f"{save_path}/CellsGraphs", f"ReductionCell{idx}")
        else:
            network.operationList[idx].draw(f"{save_path}/CellsGraphs", f"NormalCell{idx}")

    evaluateCellNetwork(network, save_path, device)

    # CONCATENATION EVALUATION
    save_path = f"DARTSResults/ArchitectureEvaluationCAT"

    model = SearchNetwork(CELLS_NUM)
    model.initialization(IMAGE_DIM, IN_CHANNELS, FIRST_CELL_CHANNELS, device)
    model.load("DARTSResults/ArchitectureSearch/BestLossArchitecture/network.pt")

    cells_list = model.genotypeCellNetworkDARTS()

    EVALUATION_CELLS_NUM = 21
    REDUCTION_POS = [math.floor(EVALUATION_CELLS_NUM / 3), math.floor(EVALUATION_CELLS_NUM / 3 * 2)]

    new_cells_list = []
    for idx, cell in enumerate(cells_list):
        if cell[0] == "reduction":
            new_cells_list.append(cell)
        else:
            new_cells_list += [cell] * (REDUCTION_POS[0] // model.reduction_indices[0])

    while len(new_cells_list) < EVALUATION_CELLS_NUM:
        new_cells_list.append(new_cells_list[-1])

    del model

    evaluateCellNetwork(network, save_path, device)



if __name__ == "__main__":
    main()
