import os
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics

from Cells import CELL_PARAMETERS
from DARTS.DARTSCell import SearchNormalCell, NoPathFound, SearchReductionCell
from DARTS.OSpace import generateOSpaceCell
from Operations import *
from Utils import AverageMeter, ProgressMeter, accuracy


class SearchNetwork(nn.Module):
    def __init__(self, cells_num):
        super(SearchNetwork, self).__init__()

        self.cells_num = cells_num

        self.save_path = None
        self.reduction_conv = None
        self.operationList = torch.nn.ModuleList()
        self.fix_chn = nn.Sequential()

        self.cells_indices = []
        self.reduction_indices = []
        self.reduction_position = [math.floor(cells_num * 1 / 3), math.floor(cells_num * 2 / 3)]

        #  Activation Function
        self.activation = nn.LeakyReLU()

        # Factorized Reduce Layers
        self.reduces_skip = nn.ModuleList()

        cell_nodes = max(CELL_PARAMETERS["cell_nodes"])
        edge_num = sum(i for i in range(cell_nodes-1, 0, -1))

        self.cell_alphas = torch.nn.Parameter(1e-3 * torch.zeros((cells_num, edge_num, len(generateOSpaceCell()))), requires_grad=True)
        self.reduction_alphas = torch.nn.Parameter(1e-3 * torch.zeros((len(self.reduction_position), edge_num, len(generateOSpaceCell()))), requires_grad=True)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)

    def initialization(self, input_dim, in_channels, first_cell_chn, device):
        self.fix_chn.append(nn.Conv2d(in_channels, first_cell_chn, 3, padding=1, bias=False))
        self.fix_chn.append(nn.BatchNorm2d(first_cell_chn))
        i = 0
        cell_nodes = max(CELL_PARAMETERS["cell_nodes"])
        while i < self.cells_num:
            try:
                if i == 0:
                    cell = SearchNormalCell(input_dim, first_cell_chn, first_cell_chn, cell_nodes, device)
                else:
                    if isinstance(self.operationList[-1], SearchNormalCell):
                        image_dim = self.operationList[-1].image_dim
                        channels = self.operationList[-1].channels
                    else:
                        image_dim = math.ceil(self.operationList[-2].image_dim / 2)
                        channels = self.operationList[-2].channels * 2
                    in_channels = channels * (cell_nodes - 1)
                    cell = SearchNormalCell(image_dim, in_channels, channels, cell_nodes, device)

                self.cells_indices.append(len(self.operationList))
                self.operationList.append(cell)

                if i in self.reduction_position:
                    self.operationList.append(SearchReductionCell(image_dim, in_channels, channels, cell_nodes, device))
                    self.reduces_skip.append(FactorizedReduce(in_channels, in_channels * 2))
                    self.reduction_indices.append(len(self.operationList) - 1)

                i += 1
            except NoPathFound:
                pass

        self.operationList.append(nn.Linear(in_channels, 10, bias=False))
        # in_dim_classifier = pow(self.operationList[-1].image_dim, 2) * channels
        # self.operationList.append(GenericMLP(in_dim_classifier, int(in_dim_classifier / 2 ** ((math.log(in_dim_classifier, 2) - math.log(8, 2)) / 2)), 10, 0))

    def forward(self, out):
        out = self.activation(self.fix_chn(out))

        delay = 0
        out = self.activation(self.operationList[0](out, self.cell_alphas[0]))
        prev_1 = torch.clone(out)

        for idx, layer in enumerate(self.operationList[1:-1], start=1):
            if idx in self.reduction_indices:
                out = self.activation(layer(out, self.reduction_alphas[delay]))
                delay += 1
            else:
                out = self.activation(layer(out, self.cell_alphas[idx - delay]))

            prev_2 = torch.clone(out)
            if out.shape != prev_1.shape:
                prev_1 = self.activation(self.reduces_skip[delay - 1](prev_1))
            out += prev_1
            out = self.activation(out)
            prev_1 = torch.clone(prev_2)

        out = self.activation(self.global_pooling(out).reshape(-1, out.shape[1]))
        return self.operationList[-1](out)
        # return self.operationList[-1](torch.flatten(batched_images, start_dim=1))

    def genotypeCellNetwork(self):
        gene_normal_cells_indices = np.argmax(F.softmax(self.cell_alphas, dim=-1).data.cpu().numpy(), axis=-1)
        gene_reduction_cells_indices = np.argmax(F.softmax(self.reduction_alphas, dim=-1).data.cpu().numpy(), axis=-1)
        O_space = generateOSpaceCell()
        delay = 0
        cells_list = []
        for cell_idx, cell in enumerate(self.operationList[:-1]):

            if isinstance(cell, SearchNormalCell):
                cells_list.append(("normal", []))
                for edge_idx in range(self.cell_alphas[cell_idx - delay].shape[0]):
                    best_op_idx = gene_normal_cells_indices[cell_idx - delay][edge_idx]
                    cells_list[cell_idx][1].append(O_space[best_op_idx])
            else:
                cells_list.append(("reduction", []))
                for edge_idx in range(self.reduction_alphas[delay].shape[0]):
                    best_op_idx = gene_reduction_cells_indices[delay][edge_idx]
                    cells_list[cell_idx][1].append(O_space[best_op_idx])
                delay += 1
        return cells_list

    def train_step(self, weights_dataloader, alpha_dataloader, weights_opt, alpha_opt, criterion, epoch, device):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.6f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(weights_dataloader),
            [batch_time, losses, top1, top5],
            prefix=f"Epoch: [{epoch}]")

        end = time()
        for idx, (batched_images, labels) in enumerate(weights_dataloader):  # tqdm(, position=0, leave=True, desc=f"Network optimization: ", colour="white", ncols=80)):

            batched_images = batched_images.to(device)
            labels = labels.to(device)

            search_images, search_labels = next(iter(alpha_dataloader))
            search_images = search_images.to(device)
            search_labels = search_labels.to(device)

            # Alpha Optimization
            alpha_opt.zero_grad()
            pred = self(search_images)
            loss = criterion(pred, search_labels)
            loss.backward()
            alpha_opt.step()

            # Weights Optimization
            weights_opt.zero_grad()
            pred = self(batched_images)
            loss = criterion(pred, labels)

            acc1, acc5 = accuracy(pred, labels, topk=(1, 5))
            losses.update(loss.item(), batched_images.size(0))
            top1.update(acc1.item(), batched_images.size(0))
            top5.update(acc5.item(), batched_images.size(0))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5)
            weights_opt.step()

            # measure elapsed time
            batch_time.update(time() - end)
            end = time()

            if idx % 100 == 0:
                progress.display(idx)

        return top1, top5, losses


    def test(self, test_dataset, criterion, batch_size, device):
        dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.6f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(dataloader),
            [batch_time, losses, top1, top5],
            prefix='Test: ')

        self.eval()
        predicted_labels = np.empty(0)

        with torch.no_grad():
            end = time()
            for idx, (batched_images, label) in enumerate(dataloader):
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

                # if idx % int(len(dataloader) / 10) == 0:
                #     progress.display(idx)

            confusion_matrix = metrics.confusion_matrix(test_dataset.labels, predicted_labels.tolist(), normalize="true")
        self.train()
        return top1.avg, top5.avg, losses.avg, confusion_matrix

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
