import math
import os
import pickle as pkl
import sys

import numpy as np
import torch
import torch.nn as nn

from CIFAR10Dataset import CIFAR10Dataset
from CellNetwork import CellNetwork

from DARTS.SearchNetwork import SearchNetwork
from PlotUtils import plot_training_statistics, plot_history


def main():
    os.chdir(os.path.dirname(sys.argv[0]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = f"DARTSResults/ArchitectureEvaluation"

    CELLS_NUM = 5
    IMAGE_DIM = 32
    IN_CHANNELS = 3
    FIRST_CELL_CHANNELS = 16

    model = SearchNetwork(CELLS_NUM)
    model.initialization(IMAGE_DIM, IN_CHANNELS, FIRST_CELL_CHANNELS, device)
    model.load("DARTSResults/ArchitectureSearch/BestAccArchitecture/network.pt")

    cells_list = model.genotypeCellNetwork()
    del model

    first_block = cells_list[:2] * 3
    second_block = cells_list[3:5] * 3
    last_block = cells_list[-1:] * 3
    new_cells_list = first_block + cells_list[2:3] + second_block + cells_list[5:6] + last_block

    network = CellNetwork()
    network.initialization(new_cells_list, 32, 3, 16, device)

    network.to(device)

    train_dataset = CIFAR10Dataset(path="../CIFAR-10", train=True, device="cpu", quantity=100)
    val_dataset = CIFAR10Dataset(path="../CIFAR-10", train=False, device="cpu", quantity=100)

    best_acc = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters())
    batch_size = 64
    torch.manual_seed(666)

    validations_top1 = []
    validations_top5 = []
    validations_loss = []
    train_losses = []

    statistics_history = {}
    statistics_history["train_top1_history"] = []
    statistics_history["train_top5_history"] = []
    statistics_history["train_losses_history"] = []

    for epoch in range(100):
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

        top1_train_acc, top5_train_acc, train_loss = network.train_step(train_dataloader, optimizer, criterion, epoch, device)
        statistics_history["train_top1_history"] += top1_train_acc.data_history
        statistics_history["train_top5_history"] += top5_train_acc.data_history
        statistics_history["train_losses_history"] += train_loss.data_history

        torch.cuda.empty_cache()
        val_top1, val_top5, val_loss, _ = network.test(val_dataset, criterion=criterion, device=device, batch_size=batch_size)

        validations_top1.append(val_top1)
        validations_top5.append(val_top5)
        validations_loss.append(val_loss)
        train_losses.append(train_loss.avg)

        print(f"Validation Test\n"
              f"Accuracy : {val_top1}\n"
              f"Loss.: {val_loss}\n")

        if best_acc < val_top1:
            best_acc = val_top1
            network.save(f"{save_path}/TrainedNet")

        plot_training_statistics(save_path, f"NetworkTraining", range(epoch + 1), train_losses, np.array(validations_top1) / 100, np.array(validations_top5) / 100, validations_loss)

        plot_history(save_path, "Top 1 Train Acc.", statistics_history["train_top1_history"], color="blueviolet")
        plot_history(save_path, "Top 5 Train Acc.", statistics_history["train_top5_history"], color="mediumorchid")
        plot_history(save_path, "Train Loss", statistics_history["train_losses_history"], color="dodgerblue")
        with open(f"{save_path}/history_dict.pkl", "wb") as dict_file:
            pkl.dump(statistics_history, dict_file)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
