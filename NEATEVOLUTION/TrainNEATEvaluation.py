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
from EvaluateCellNetwork import evaluateCellNetwork
from NEATEVOLUTION.NEATEvolution import NEATEvolution
from PlotUtils import plot_training_statistics, plot_history


def main():
    os.chdir(os.path.dirname(sys.argv[0]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = f"NEATEvolutionResults/Evaluation"
    os.makedirs(save_path, exist_ok=True)

    CELLS_NUM = 7
    CELL_NODES = 5

    POPULATION_FILENAME = "population09012023_1311"

    evolution = NEATEvolution(CELLS_NUM, CELL_NODES)
    evolution.loadPopulation("Populations", POPULATION_FILENAME)

    best_net_list = evolution.bestNetwork(21)

    evaluateCellNetwork(best_net_list, save_path, device)


if __name__ == "__main__":
    main()
