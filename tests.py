import os
import sys

import numpy as np
import torch

from CIFAR10Dataset import CIFAR10Dataset, data_reverse_transform_cifar10
from CellNetwork import CellNetwork
from NEATEVOLUTION.NEATEvolution import NEATEvolution


def main():
    # os.chdir(os.path.dirname(sys.argv[0]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    IMAGE_DIM = 32
    IN_CHANNELS = 3
    FIRST_CELL_CHANNELS = 36
    CELLS_NUM = 7
    CELL_NODES = 5

    POPULATION_FILENAME = "population10012023_0126"

    evolution = NEATEvolution(CELLS_NUM, CELL_NODES)
    evolution.loadPopulation("NEATEVOLUTION/Populations", POPULATION_FILENAME)

    best_net_list = evolution.bestNetwork(21)

    inv_trans = data_reverse_transform_cifar10()

    val_dataset = CIFAR10Dataset(path="../CIFAR-10", train=False, device="cpu", quantity=100)
    val_dataset.to(device)

    network = CellNetwork()
    network.initialization(best_net_list, IMAGE_DIM, IN_CHANNELS, FIRST_CELL_CHANNELS)

    network.load("U:\Magistrale\Machine Learning/NeuralArchitectureSearch/NEATEVOLUTION/NEATEvolutionResults\Evaluation\TrainedNet/network.pt")

    network.to(device)

    frogs = torch.nonzero(val_dataset.labels == 4)
    for frog in frogs:
        image, label = val_dataset[frog]

        pred = network(image.view(1, 3, 32, 32))
        pred_class = np.argmax(pred.detach().cpu().numpy()[0])
        image = inv_trans(image)
        print(label, pred_class)
        print(val_dataset.classes[label], val_dataset.classes[pred_class])
        visualizeCIFARImage(image, val_dataset.classes[label], val_dataset.classes[pred_class])


def visualizeCIFARImage(rgb_values, label, predicted_label):
    import matplotlib.pyplot as plt
    # Reshape the image
    rgb_values = torch.swapaxes(rgb_values, 0, 1)
    rgb_values = torch.swapaxes(rgb_values, 1, 2)
    image = rgb_values.detach().cpu().numpy()
    plt.title(f"Label: {label}, Predicted: {predicted_label}")
    # Display the image
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    main()
