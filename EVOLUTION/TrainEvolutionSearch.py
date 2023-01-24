import os
import sys

from Evolution import Evolution


CELL_NUMBER = 7
IMAGE_DIM = 32
CHANNELS = 3


def main():
    os.chdir(os.path.dirname(sys.argv[0]))

    evolution = Evolution(cells_num=CELL_NUMBER, cell_nodes=5, image_dim=IMAGE_DIM, in_channels=CHANNELS, first_cell_channels=16, population=20)
    evolution.evolveNetworkPopulation(generations=36, to_kill=0.7, decision_mode="loss")


if __name__ == "__main__":
    main()
