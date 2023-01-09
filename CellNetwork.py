import os
from time import time
import pickle as pkl
import numpy as np
from sklearn import metrics

from Cells import NormalCell, ReductionCell
from Operations import *
from Utils import AverageMeter, ProgressMeter, accuracy


class CellNetwork(nn.Module):
    def __init__(self):
        super(CellNetwork, self).__init__()

        self.save_path = None
        self.reduction_conv = None
        self.operationList = torch.nn.ModuleList()
        self.fix_chn = nn.Sequential()

        self.cells_indices = []
        self.reduction_indices = []

        #  Activation Function
        self.activation = nn.LeakyReLU()

        # Factorized Reduce Layers
        self.reduces_skip = nn.ModuleList()

        self.global_pooling = nn.AdaptiveAvgPool2d(1)

    def initialization(self, network_cells, input_dim, in_channels, first_cell_chn):
        self.fix_chn.append(nn.Conv2d(in_channels, first_cell_chn, 3, padding=1, bias=False))
        self.fix_chn.append(nn.BatchNorm2d(first_cell_chn))

        step = 1
        edge_num = len(network_cells[0][1])
        while edge_num > 0:
            edge_num -= step
            step += 1

        cell_nodes = step
        for cell_idx, cell_tuple in enumerate(network_cells):
            if cell_tuple[0] == "normal" and cell_idx == 0:
                self.operationList.append(NormalCell(cell_tuple[1], input_dim, first_cell_chn, first_cell_chn, cell_nodes))
                self.cells_indices.append(len(self.operationList) - 1)
            else:
                if isinstance(self.operationList[-1], NormalCell):
                    image_dim = self.operationList[-1].image_dim
                    channels = self.operationList[-1].channels
                else:
                    image_dim = math.ceil(self.operationList[-2].image_dim / 2)
                    channels = self.operationList[-2].channels * 2
                in_channels = channels * (cell_nodes - 1)

                if cell_tuple[0] == "normal":
                    self.operationList.append(NormalCell(cell_tuple[1], image_dim, in_channels, channels, cell_nodes))
                    self.cells_indices.append(len(self.operationList) - 1)
                else:
                    self.operationList.append(ReductionCell(cell_tuple[1], image_dim, in_channels, channels, cell_nodes))
                    self.reduces_skip.append(FactorizedReduce(in_channels, in_channels * 2))
                    self.reduction_indices.append(len(self.operationList) - 1)

        self.operationList.append(nn.Linear(in_channels, 10, bias=False))
        # in_dim_classifier = pow(self.operationList[-1].image_dim, 2) * channels
        # self.operationList.append(GenericMLP(in_dim_classifier, int(in_dim_classifier / 2 ** ((math.log(in_dim_classifier, 2) - math.log(8, 2)) / 2)), 10, 0))

    def forward(self, out):
        out = self.activation(self.fix_chn(out))

        delay = 0
        out = self.activation(self.operationList[0](out))
        prev_1 = torch.clone(out)

        for idx, layer in enumerate(self.operationList[1:-1], start=1):
            if idx in self.reduction_indices:
                delay += 1
            out = self.activation(layer(out))
            prev_2 = torch.clone(out)
            if out.shape != prev_1.shape:
                prev_1 = self.activation(self.reduces_skip[delay - 1](prev_1))
            out += prev_1
            out = self.activation(out)
            prev_1 = torch.clone(prev_2)
        out = self.global_pooling(out).view(out.size(0), -1)
        return self.operationList[-1](out)
        # return self.operationList[-1](torch.flatten(batched_images, start_dim=1))

    def train_step(self, train_dataloader, optimizer, criterion, epoch, device, verbose=1):
        # switch to train mode
        self.train()

        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.6f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(train_dataloader),
            [batch_time, losses, top1, top5],
            prefix=f"Epoch: [{epoch}]")

        end = time()
        for idx, (batched_images, labels) in enumerate(train_dataloader):  # tqdm(, position=0, leave=True, desc=f"Network optimization: ", colour="white", ncols=80)):
            batched_images = batched_images.to(device)
            labels = labels.to(device)

            # Weights Optimization
            optimizer.zero_grad()
            pred = self(batched_images)
            loss = criterion(pred, labels)

            acc1, acc5 = accuracy(pred, labels, topk=(1, 5))
            losses.update(loss.item(), batched_images.size(0))
            top1.update(acc1.item(), batched_images.size(0))
            top5.update(acc5.item(), batched_images.size(0))

            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time() - end)
            end = time()

            if idx % 100 == 0 and verbose > 0:
                progress.display(idx)

        return top1, top5, losses

    def test(self, test_dataset, criterion, batch_size, device):
        test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.6f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        self.eval()
        predicted_labels = np.empty(0)

        with torch.no_grad():
            end = time()
            for idx, (batched_images, label) in enumerate(test_dataloader):
                batched_images = batched_images.to(device)
                label = label.to(device)
                pred = self(batched_images)
                loss = criterion(pred, label)

                acc1, acc5 = accuracy(pred, label, topk=(1, 5))
                predicted_labels = np.hstack((predicted_labels, pred.cpu().argmax(dim=1)))
                losses.update(loss.item(), batched_images.size(0))
                top1.update(acc1.item(), batched_images.size(0))
                top5.update(acc5.item(), batched_images.size(0))

                batch_time.update(time() - end)
                end = time()

            confusion_matrix = metrics.confusion_matrix(test_dataset.labels, predicted_labels.tolist(), normalize="true")
        self.train()
        return top1.avg, top5.avg, losses.avg, confusion_matrix

    def save(self, path, filename="network.pt"):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/" + filename)

    def save_as_pkl(self, path, filename="network.pkl"):
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/{filename}", "wb") as net_file:
            pkl.dump(self, net_file)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
