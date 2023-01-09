from time import time
import pickle as pkl

import numpy as np
import torch
from torch import nn

from CIFAR10Dataset import CIFAR10Dataset
from CellNetwork import CellNetwork
from PlotUtils import plot_training_statistics, plot_history


def evaluateCellNetwork(cell_list, save_path, device):
    # Training Parameters
    BATCH_SIZE = 96
    EVALUATION_EPOCHS = 60

    # Input Parameters
    IMAGE_DIM = 32
    IN_CHANNELS = 3
    FIRST_CELL_CHANNELS = 36

    network = CellNetwork()
    network.initialization(cell_list, IMAGE_DIM, IN_CHANNELS, FIRST_CELL_CHANNELS)

    network.operationList[0].draw(f"{save_path}/BestLossCell", "BestNormalCell")
    network.operationList[network.reduction_indices[0]].draw(f"{save_path}/BestLossCell", "BestReductionCell")

    network.to(device)

    train_dataset = CIFAR10Dataset(path="../CIFAR-10", train=True, device="cpu", quantity=100)
    val_dataset = CIFAR10Dataset(path="../CIFAR-10", train=False, device="cpu", quantity=100)

    best_loss = 1000000000000
    best_loss_acc = None

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(network.parameters())

    torch.manual_seed(666)

    validations_top1 = []
    validations_top5 = []
    validations_loss = []
    train_losses = []

    statistics_history = {}
    statistics_history["train_top1_history"] = []
    statistics_history["train_top5_history"] = []
    statistics_history["train_losses_history"] = []
    training_times = []
    start_time = time()

    for epoch in range(EVALUATION_EPOCHS):
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

        top1_train_acc, top5_train_acc, train_loss = network.train_step(train_dataloader, optimizer, criterion, epoch, device)

        training_times.append((time() - start_time) / 60)
        statistics_history["train_top1_history"] += top1_train_acc.data_history
        statistics_history["train_top5_history"] += top5_train_acc.data_history
        statistics_history["train_losses_history"] += train_loss.data_history

        val_top1, val_top5, val_loss, _ = network.test(val_dataset, criterion=criterion, device=device, batch_size=BATCH_SIZE)

        validations_top1.append(val_top1)
        validations_top5.append(val_top5)
        validations_loss.append(val_loss)
        train_losses.append(train_loss.avg)

        print(f"Validation Test\n"
              f"Accuracy : {val_top1}\n"
              f"Loss.: {val_loss}\n")

        if best_loss > val_loss:
            best_loss = val_loss
            best_loss_acc = val_top1
            network.save(f"{save_path}/TrainedNet")

        plot_training_statistics(save_path, f"NetworkTrainingStatistics", range(epoch + 1), train_losses, np.array(validations_top1) / 100, np.array(validations_top5) / 100, validations_loss, title=f"Best Loss {round(best_loss, 2)} and acc. {best_loss_acc}")
        plot_training_statistics(save_path, f"NetworkTrainingTime", training_times, train_losses, np.array(validations_top1) / 100, np.array(validations_top5) / 100, validations_loss, title=f"Best Loss {round(best_loss, 2)} and acc. {best_loss_acc}", x_axis_label="Elapsed Time (Min)")

        plot_history(save_path, "Top 1 Train Acc.", statistics_history["train_top1_history"], color="blueviolet")
        plot_history(save_path, "Top 5 Train Acc.", statistics_history["train_top5_history"], color="mediumorchid")
        plot_history(save_path, "Train Loss", statistics_history["train_losses_history"], color="dodgerblue")
        with open(f"{save_path}/history_dict.pkl", "wb") as dict_file:
            pkl.dump(statistics_history, dict_file)
        torch.cuda.empty_cache()
