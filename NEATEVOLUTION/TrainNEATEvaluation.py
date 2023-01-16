import os
import sys

import torch

from EvaluateCellNetwork import evaluateCellNetwork
from NEATEVOLUTION.NEATEvolution import NEATEvolution


def main():
    os.chdir(os.path.dirname(sys.argv[0]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = f"NEATEvolutionResults/Evaluation"
    os.makedirs(save_path, exist_ok=True)

    CELLS_NUM = 7
    CELL_NODES = 5

    POPULATION_FILENAME = "population13012023_0109"

    evolution = NEATEvolution(CELLS_NUM, CELL_NODES)
    evolution.loadPopulation("Populations", POPULATION_FILENAME)

    best_net_list = evolution.bestNetwork(21, 1, 1)

    evaluateCellNetwork(best_net_list, save_path, device)


if __name__ == "__main__":
    main()
