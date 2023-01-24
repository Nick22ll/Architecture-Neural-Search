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
from PlotUtils import plot_training_statistics, plot_history


def main():
    os.chdir(os.path.dirname(sys.argv[0]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = f"DARTSResults/CellSearch"

    CELLS_NUM = 7
    IMAGE_DIM = 32
    IN_CHANNELS = 3
    FIRST_CELL_CHANNELS = 16

    model = SearchCellNetwork(CELLS_NUM)
    model.initialization(IMAGE_DIM, IN_CHANNELS, FIRST_CELL_CHANNELS, device)
    model.to(device)

    count_parameters_in_MB(model)

    train_dataset = CIFAR10Dataset(path="../CIFAR-10", train=True, device="cpu", quantity=100)
    train_dataset, val_dataset = train_dataset.split(50)

    criterion = nn.CrossEntropyLoss()
    batch_size = 64
    epochs = 60

    # switch to train mode
    model.train()
    best_loss = 100000000
    best_loss_acc = None

    alpha_parameters = []
    weights_parameters = []

    for n, p in model.named_parameters():
        if n in ["cell_alphas", "reduction_alphas"]:
            alpha_parameters.append(p)
        else:
            weights_parameters.append(p)

    weights_optimizer = torch.optim.SGD(weights_parameters, lr=0.025, momentum=0.9, weight_decay=3 * 1e-4)  # torch.optim.Adam(weights_parameters, lr=0.025, amsgrad=True)
    alpha_optimizer = torch.optim.Adam(alpha_parameters, lr=3 * 1e-4, weight_decay=1e-3, amsgrad=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        weights_optimizer, epochs, eta_min=0.001)

    torch.manual_seed(666)
    validations_top1 = []
    validations_top5 = []
    validations_loss = []
    train_losses = []

    statistics_history = {"train_top1_history": [], "train_top5_history": [], "train_losses_history": []}

    train_start = time()
    epoch_times = []

    for epoch in range(epochs):

        alpha_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        weights_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

        top1_train_acc, top5_train_acc, train_loss = model.train_step(weights_dataloader, alpha_dataloader, weights_optimizer, alpha_optimizer, criterion, epoch, device)

        epoch_times.append((time() - train_start) / 60)

        statistics_history["train_top1_history"] += top1_train_acc.data_history
        statistics_history["train_top5_history"] += top5_train_acc.data_history
        statistics_history["train_losses_history"] += train_loss.data_history

        scheduler.step()

        torch.cuda.empty_cache()
        val_top1, val_top5, val_loss, _ = model.test(test_dataset=val_dataset, criterion=criterion, batch_size=batch_size, device=device)

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
            model.save(f"{save_path}/BestLossCell")

        plot_training_statistics(save_path, f"CellTraining", range(epoch + 1), train_losses, np.array(validations_top1) / 100, np.array(validations_top5) / 100, validations_loss, title=f"Best Loss {round(best_loss, 2)} and Acc. {best_loss_acc}")
        plot_training_statistics(save_path, f"CellTrainingTimes", epoch_times, train_losses, np.array(validations_top1) / 100, np.array(validations_top5) / 100, validations_loss, title=f"Best Loss {round(best_loss, 2)} and Acc. {best_loss_acc}", x_axis_label="Elapsed Time (Min)")

        plot_history(save_path, "Top 1 Train Acc.", statistics_history["train_top1_history"])
        plot_history(save_path, "Top 5 Train Acc.", statistics_history["train_top5_history"])
        plot_history(save_path, "Train Loss", statistics_history["train_losses_history"])
        with open(f"{save_path}/history_dict.pkl", "wb") as dict_file:
            pkl.dump(statistics_history, dict_file)

        torch.cuda.empty_cache()


def count_parameters_in_MB(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('model size: {:.3f}MB'.format(size_all_mb))


if __name__ == "__main__":
    main()
