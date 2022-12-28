import warnings

import dgl
import numpy as np
import torch.nn.functional as F

from Cells import generate_nodes, CELL_PARAMETERS
from DARTS.OSpace import generateOSpaceCell, generateOSpaceCellChannels
from Operations import *


class NoPathFound(Exception):
    pass


class MixedOp(nn.Module):
    def __init__(self, channels, device):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()

        stride = 1
        for primitive in generateOSpaceCell():
            split_O = primitive.split("$")
            op_parameters = {}
            for i in range(0, len(split_O) - 1, 2):
                op_parameters[split_O[i]] = int(split_O[i + 1])

            if op_parameters["operation"] in [3]:  # if convolution
                op = nn.Sequential(OPERATIONS[op_parameters["operation"]](channels, channels, op_parameters["kernel_size"], stride=stride, padding=math.floor(op_parameters["kernel_size"] / 2), bias=False, device=device),
                                   nn.BatchNorm2d(channels, affine=False, device=device))
            elif op_parameters["operation"] in [1, 2]:
                op = nn.Sequential(OPERATIONS[op_parameters["operation"]](op_parameters["kernel_size"], stride=stride, padding=math.floor(op_parameters["kernel_size"] / 2)),
                                   nn.BatchNorm2d(channels, affine=False, device=device))
            elif op_parameters["operation"] in [0]:
                op = ZeroOp()
            elif op_parameters["operation"] in [4]:
                op = OPERATIONS[op_parameters["operation"]]()
            else:
                raise ()
            self._ops.append(op)

    def forward(self, x, alpha):
        weights = F.softmax(alpha, dim=-1)
        return sum(w * op(x) for w, op in zip(weights, self._ops, strict=True))


# class MixedOpChan(nn.Module):
#     def __init__(self, input_dim, input_channels, device):
#         super(MixedOpChan, self).__init__()
#         self.image_dim = input_dim
#         self._ops = nn.ModuleList()
#         for primitive in generateOSpaceCellChannels():
#             split_O = primitive.split("$")
#             op_parameters = {}
#             for i in range(0, len(split_O) - 1, 2):
#                 op_parameters[split_O[i]] = int(split_O[i + 1])
#             self._ops.append(SearchNormalCell(input_dim, input_channels, op_parameters["cell_nodes"], op_parameters["cell_channels"], device))
#
#     def forward(self, x, channel_alpha, cell_alpha):
#         weights = F.softmax(channel_alpha, dim=-1)
#         results = []
#         for idx, op in enumerate(self._ops):
#             res = op(x, cell_alpha)
#             # TILING VERSION  results.append(res.tile((1, int(max(CELL_PARAMETERS["cell_channels"]) / res.shape[1]), 1, 1)))
#             results.append(torch.cat((res, torch.zeros_like(res).tile((1, int(max(CELL_PARAMETERS["cell_channels"]) / res.shape[1]) - 1, 1, 1))), dim=1))
#         return sum(w * r for w, r in zip(weights, results, strict=True))


class SearchNormalCell(nn.Module, dgl.DGLGraph):
    def __init__(self, input_dim, in_channels, channels, cell_nodes, device):
        super(SearchNormalCell, self).__init__()

        self.image_dim = input_dim
        self.channels = channels

        self.activation = nn.LeakyReLU()

        self.chn_fix = ChannelFixer(in_channels, channels, 1, 1, 0, affine=False)

        # nodes generation
        start_nodes, end_nodes = generate_nodes(cell_nodes)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            dgl.DGLGraph.__init__(self, (start_nodes, end_nodes))

        self.operations_list = torch.nn.ModuleList()
        for edge_idx in range(self.num_edges()):
            self.operations_list.append(MixedOp(channels=self.channels, device=device))

    def forward(self, batched_images, alphas):
        end_nodes = self.edges()[1]
        feature_states = {}
        feature_states[0] = self.chn_fix(batched_images)
        for node_id in range(1, self.num_nodes()):
            edge_indices = np.where(end_nodes == node_id)[0]
            updated_feats = sum(self.operations_list[edge_idx](feature_states[self.edges()[0][edge_idx].item()], alphas[edge_idx]) for edge_idx in edge_indices)
            feature_states[node_id] = updated_feats
        edge_indices = np.where(end_nodes == self.num_nodes() - 1)[0]
        return torch.cat([self.operations_list[edge_idx](feature_states[self.edges()[0][edge_idx].item()], alphas[edge_idx]) for edge_idx in edge_indices], dim=1)

    def to(self, device):
        nn.Module.to(self, device=device)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class SearchChnCell(nn.Module, dgl.DGLGraph):
    def __init__(self, input_dim, in_channels, cell_nodes, cell_channels, device):
        super(SearchChnCell, self).__init__()

        self.image_dim = input_dim
        self.channels = cell_channels
        # nodes generation
        start_nodes, end_nodes = generate_nodes(cell_nodes)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            dgl.DGLGraph.__init__(self, (start_nodes, end_nodes))

        self.chn_fix_conv = nn.Conv2d(in_channels, self.channels, kernel_size=1, stride=1, dilation=1, bias=False)

        self.operations_list = torch.nn.ModuleList()
        for edge_idx in range(self.num_edges()):
            self.operations_list.append(MixedOp(channels=self.channels, device=device))

    def forward(self, batched_images, alphas):
        end_nodes = self.edges()[1]
        feature_states = {}
        feature_states[0] = self.chn_fix_conv(batched_images)
        for node_id in range(1, self.num_nodes() - 1):
            edge_indices = np.where(end_nodes == node_id)[0]
            updated_feats = sum(self.activation(self.operations_list[edge_idx](feature_states[self.edges()[0][edge_idx].item()], alphas[edge_idx])) for edge_idx in edge_indices)
            feature_states[node_id] = self.activation(updated_feats)
        return feature_states[self.num_nodes() - 1]

    def to(self, device):
        nn.Module.to(self, device=device)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class ReductionMixedOp(nn.Module):
    def __init__(self, channels, device):
        super(ReductionMixedOp, self).__init__()
        self._ops = nn.ModuleList()
        stride = 2

        for primitive in generateOSpaceCell():
            split_O = primitive.split("$")
            op_parameters = {}
            for i in range(0, len(split_O) - 1, 2):
                op_parameters[split_O[i]] = int(split_O[i + 1])

            if op_parameters["operation"] in [3]:  # if convolution
                op = nn.Sequential(OPERATIONS[op_parameters["operation"]](channels, channels, op_parameters["kernel_size"], stride=stride, padding=math.floor(op_parameters["kernel_size"] / 2), bias=False, device=device),
                                   nn.BatchNorm2d(channels, affine=False, device=device))
            elif op_parameters["operation"] in [1, 2]:
                op = nn.Sequential(OPERATIONS[op_parameters["operation"]](op_parameters["kernel_size"], stride=stride, padding=math.floor(op_parameters["kernel_size"] / 2)),
                                   nn.BatchNorm2d(channels, affine=False, device=device))
            elif op_parameters["operation"] in [0]:
                op = ZeroOp(stride)
            elif op_parameters["operation"] in [4]:
                op = FactorizedReduce(channels, channels, affine=False)
            else:
                raise ()
            self._ops.append(op)

    def forward(self, x, alpha):
        weights = F.softmax(alpha, dim=-1)
        return sum(w * op(x) for w, op in zip(weights, self._ops, strict=True))


class SearchReductionCell(nn.Module, dgl.DGLGraph):
    def __init__(self, input_dim, in_channels, channels, cell_nodes, device):
        super(SearchReductionCell, self).__init__()

        self.image_dim = input_dim // 2
        self.channels = channels * 2

        self.activation = nn.LeakyReLU()

        self.chn_fix = ChannelFixer(in_channels, self.channels, 1, 1, 0, affine=False)

        # nodes generation
        start_nodes, end_nodes = generate_nodes(cell_nodes)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            dgl.DGLGraph.__init__(self, (start_nodes, end_nodes))

        self.operations_list = torch.nn.ModuleList()

        for edge_idx in range(self.num_nodes() - 1):
            self.operations_list.append(ReductionMixedOp(channels=self.channels, device=device))

        for edge_idx in range(self.num_nodes() - 1, self.num_edges()):
            self.operations_list.append(MixedOp(channels=self.channels, device=device))

    def forward(self, batched_images, alphas):
        end_nodes = self.edges()[1]
        feature_states = {}
        feature_states[0] = self.chn_fix(batched_images)
        for node_id in range(1, self.num_nodes()):
            edge_indices = np.where(end_nodes == node_id)[0]
            updated_feats = sum(self.operations_list[edge_idx](feature_states[self.edges()[0][edge_idx].item()], alphas[edge_idx]) for edge_idx in edge_indices)
            feature_states[node_id] = updated_feats
        edge_indices = np.where(end_nodes == self.num_nodes() - 1)[0]
        return torch.cat([self.operations_list[edge_idx](feature_states[self.edges()[0][edge_idx].item()], alphas[edge_idx]) for edge_idx in edge_indices], dim=1)


class CellChannels(nn.Module):
    def __init__(self, in_channels, device):
        super(CellChannels, self).__init__()

        self.reduction_op = ReductionMixedOp(channels=in_channels, device=device)

    def forward(self, batched_images, alpha):
        return self.reduction_op(batched_images, alpha)
