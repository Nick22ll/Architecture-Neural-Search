import os
from time import time

import numpy as np
import torch.nn.functional as F
from sklearn import metrics

from Cells import CELL_PARAMETERS
from DARTS.DARTSCell import SearchNormalCell, NoPathFound, SearchReductionCell
from DARTS.OSpace import generateOSpaceCell
from Operations import *
from Utils import AverageMeter, ProgressMeter, accuracy


class SearchCellNetwork(nn.Module):
    def __init__(self, cells_num):
        super(SearchCellNetwork, self).__init__()

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
        edge_num = sum(i for i in range(cell_nodes - 1, 0, -1))

        self.normal_alphas = torch.nn.Parameter(1e-3 * torch.randn((edge_num, len(generateOSpaceCell()))), requires_grad=True)
        self.reduction_alphas = torch.nn.Parameter(1e-3 * torch.randn((edge_num, len(generateOSpaceCell()))), requires_grad=True)

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

                if i + 1 in self.reduction_position:
                    self.operationList.append(SearchReductionCell(image_dim, in_channels, channels, cell_nodes, device))
                    self.reduces_skip.append(FactorizedReduce(in_channels, in_channels * 2))
                    self.reduction_indices.append(len(self.operationList) - 1)

                i += 1
            except NoPathFound:
                pass

        self.operationList.append(nn.Linear(in_channels, 10, bias=False))

    def forward(self, out):
        out = self.activation(self.fix_chn(out))

        delay = 0
        out = self.activation(self.operationList[0](out, self.normal_alphas))
        prev_1 = torch.clone(out)

        for idx, layer in enumerate(self.operationList[1:-1], start=1):
            if idx in self.reduction_indices:
                out = self.activation(layer(out, self.reduction_alphas))
                delay += 1
            else:
                out = self.activation(layer(out, self.normal_alphas))

            prev_2 = torch.clone(out)
            if out.shape != prev_1.shape:
                prev_1 = self.activation(self.reduces_skip[delay - 1](prev_1))
            out += prev_1
            out = self.activation(out)
            prev_1 = torch.clone(prev_2)

        out = self.global_pooling(out).view(out.size(0), -1)
        return self.operationList[-1](out)


    def genotypeCellNetworkNoZero(self):
        gene_normal_cells_indices = np.argmax(F.softmax(self.normal_alphas, dim=-1).data.cpu().numpy(), axis=-1)
        gene_reduction_cells_indices = np.argmax(F.softmax(self.reduction_alphas, dim=-1).data.cpu().numpy(), axis=-1)
        O_space = generateOSpaceCell()
        delay = 0
        cells_list = []
        for cell_idx, cell in enumerate(self.operationList[:-1]):
            if isinstance(cell, SearchNormalCell):
                cells_list.append(("normal", []))
                for edge_idx in range(self.normal_alphas.shape[0]):
                    best_op_idx = gene_normal_cells_indices[cell_idx - delay][edge_idx]
                    cells_list[cell_idx][1].append(O_space[best_op_idx])
            else:
                cells_list.append(("reduction", []))
                for edge_idx in range(self.reduction_alphas.shape[0]):
                    best_op_idx = gene_reduction_cells_indices[delay][edge_idx]
                    cells_list[cell_idx][1].append(O_space[best_op_idx])
                delay += 1
        return cells_list

    def genotypeCellNetworkDARTS(self):
        O_space = generateOSpaceCell()
        op_zero_idx = O_space.index("operation$0$")

        no_zero_normal_alphas = torch.cat((self.normal_alphas[:, :op_zero_idx], self.normal_alphas[:, op_zero_idx + 1:]), dim=-1)
        no_zero_reduction_alphas = torch.cat((self.reduction_alphas[:, :op_zero_idx], self.reduction_alphas[:, op_zero_idx + 1:]), dim=-1)

        O_space.pop(op_zero_idx)

        top_normal_operations, top_normal_operation_indices = torch.sort(F.softmax(no_zero_normal_alphas, dim=-1), dim=-1, descending=True)
        top_normal_operations, top_normal_operation_indices = top_normal_operations[:, 0], top_normal_operation_indices[:, 0]
        top_reduction_cells, top_reduction_cells_indices = torch.sort(F.softmax(no_zero_reduction_alphas, dim=-1), dim=-1, descending=True)
        top_reduction_cells, top_reduction_cells_indices = top_reduction_cells[:, 0], top_reduction_cells_indices[:, 0]

        delay = 0
        cells_list = []
        for cell_idx, cell in enumerate(self.operationList[:-1]):
            if isinstance(cell, SearchNormalCell):
                cells_list.append(("normal", ["operation$0$"] * len(cell.edges()[0])))
                start_nodes = cell.edges()[0]
                for node_id in range(0, cell.num_nodes() - 1):
                    edge_indices = np.where(start_nodes == node_id)[0]
                    _, sorted_edges_indices = torch.sort(top_normal_operations[edge_indices], descending=True, dim=-1)

                    for edge_idx in sorted_edges_indices[:2]:
                        cells_list[cell_idx][1][edge_indices[edge_idx]] = O_space[top_normal_operation_indices[edge_indices[edge_idx]]]

            else:
                cells_list.append(("reduction", ["operation$0$"] * len(cell.edges()[0])))
                start_nodes = cell.edges()[0]
                for node_id in range(0, cell.num_nodes() - 1):
                    edge_indices = np.where(start_nodes == node_id)[0]
                    _, sorted_edges_indices = torch.sort(top_reduction_cells[edge_indices], descending=True, dim=-1)
                    for edge_idx in sorted_edges_indices[:2]:
                        cells_list[cell_idx][1][edge_indices[edge_idx]] = O_space[top_reduction_cells_indices[edge_indices[edge_idx]]]

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
