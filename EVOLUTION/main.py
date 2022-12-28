import os
import sys

from matplotlib import pyplot as plt

from Evolution import Evolution


def pixel_histogram(np_image):
    plt.hist(np_image.ravel(), bins=50, density=True)
    plt.xlabel("pixel values")
    plt.ylabel("relative frequency")
    plt.title("distribution of pixels")
    plt.show()

CELL_NUMBER = 5
IMAGE_DIM = 32
CHANNELS = 3

def main():
    os.chdir(os.path.dirname(sys.argv[0]))

    # import warnings
    # warnings.filterwarnings("error")

    evolution = Evolution(cells_num=CELL_NUMBER, cell_nodes=5, image_dim=IMAGE_DIM, in_channels=CHANNELS, first_cell_channels=16, population=20)
    evolution.evolveNetworkPopulation(generations=72, to_kill=0.7)


if __name__ == "__main__":
    main()
