import warnings
import graphviz as gv
import dgl
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from Operations import *

CELL_PARAMETERS = {"cell_nodes": [5]}  # , "cell_channels": [8, 16]
EDGE_PARAMETERS = {"operation": [0, 1, 2, 3, 4], "kernel_size": [3, 5]}  # , "padding": [0, 1], "stride": [1], "dilation": [1], "channels": [8]


def generateNodes(num_nodes):
    num_nodes = int(num_nodes)
    start_nodes = np.array([i for i in range(num_nodes) for _ in range(i + 1, num_nodes)])
    end_nodes = np.array([j for i in range(num_nodes) for j in range(i + 1, num_nodes)])
    return start_nodes, end_nodes


class NormalCell(nn.Module, dgl.DGLGraph):
    def __init__(self, edge_operations, input_dim, in_channels, channels, cell_nodes):
        super(NormalCell, self).__init__()

        self.image_dim = input_dim
        self.channels = channels

        self.activation = nn.LeakyReLU()

        self.chn_fix = ChannelFixer(in_channels, channels, 1, 1, 0, affine=False)

        # nodes generation
        start_nodes, end_nodes = generateNodes(cell_nodes)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            dgl.DGLGraph.__init__(self, (start_nodes, end_nodes))

        self.operations_list = torch.nn.ModuleList()
        self.edge_operations_dict = {}
        self.edge_operations = edge_operations
        for edge_idx in range(self.num_edges()):
            split_O = edge_operations[edge_idx].split("$")
            self.edge_operations_dict[edge_idx] = {}
            for i in range(0, len(split_O) - 1, 2):
                self.edge_operations_dict[edge_idx][split_O[i]] = int(split_O[i + 1])

            if self.edge_operations_dict[edge_idx]["operation"] in [3]:  # if convolution
                op = nn.Sequential(OPERATIONS[self.edge_operations_dict[edge_idx]["operation"]](channels, channels, self.edge_operations_dict[edge_idx].get("kernel_size", 3), stride=1, padding=math.floor(self.edge_operations_dict[edge_idx].get("kernel_size", 3) / 2), bias=False),
                                   nn.BatchNorm2d(self.channels, affine=False))
            elif self.edge_operations_dict[edge_idx]["operation"] in [1, 2]:
                op = nn.Sequential(OPERATIONS[self.edge_operations_dict[edge_idx]["operation"]](self.edge_operations_dict[edge_idx].get("kernel_size", 3), stride=1, padding=math.floor(self.edge_operations_dict[edge_idx].get("kernel_size", 3) / 2)),
                                   nn.BatchNorm2d(self.channels, affine=False))
            elif self.edge_operations_dict[edge_idx]["operation"] in [0]:
                op = ZeroOp()
            elif self.edge_operations_dict[edge_idx]["operation"] in [4]:
                op = OPERATIONS[self.edge_operations_dict[edge_idx]["operation"]]()
            else:
                raise ()
            self.operations_list.append(op)

    def forward(self, batched_images):
        end_nodes = self.edges()[1]
        feature_states = {}
        feature_states[0] = self.chn_fix(batched_images)
        for node_id in range(1, self.num_nodes()):
            edge_indices = np.where(end_nodes == node_id)[0]
            updated_feats = sum(self.operations_list[edge_idx](feature_states[self.edges()[0][edge_idx].item()]) for edge_idx in edge_indices)
            feature_states[node_id] = updated_feats
        edge_indices = np.where(end_nodes == self.num_nodes() - 1)[0]
        return torch.cat([self.operations_list[edge_idx](feature_states[self.edges()[0][edge_idx].item()]) for edge_idx in edge_indices], dim=1)

    def to(self, device):
        nn.Module.to(self, device=device)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def draw(self, path, filename):
        h = gv.Digraph(edge_attr=dict(fontsize='20', ),
                       node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='1', width='1', penwidth='2'),
                       engine="dot")
        # h.graph_attr['splines'] = 'ortho'
        h.graph_attr['ranksep'] = '1 equally'
        h.graph_attr['rankdir'] = 'TB'

        for idx in self.nodes():
            if idx == 0:
                h.node(str(idx.item()), label=str(idx.item()), fillcolor='darkseagreen2')
            elif idx == len(self.nodes())-1:
                h.node(str(idx.item()), label=str(idx.item()), fillcolor='lightblue')
            else:
                h.node(str(idx.item()), label=str(idx.item()), fillcolor='palegoldenrod')
        for idx in range(len(self.edges()[0])):
            if self.edge_operations_dict[idx]["operation"] not in [0, 4]:
                h.edge(str(self.edges()[0][idx].item()), str(self.edges()[1][idx].item()), label=f"{OPERATIONS[self.edge_operations_dict[idx]['operation']].__name__}{self.edge_operations_dict.get('kernel_size', 3)}x{self.edge_operations_dict.get('kernel_size', 3)}")
            elif self.edge_operations_dict[idx]["operation"] in [4]:
                h.edge(str(self.edges()[0][idx].item()), str(self.edges()[1][idx].item()), label=f"{OPERATIONS[self.edge_operations_dict[idx]['operation']].__name__}")

        fileformat = 'png'
        h.render(path + "/" + filename, format=fileformat, view=False, cleanup=True)



class ReductionCell(nn.Module, dgl.DGLGraph):
    def __init__(self, edge_operations, input_dim, in_channels, channels, cell_nodes):
        super(ReductionCell, self).__init__()

        self.image_dim = input_dim // 2
        self.channels = channels * 2

        self.activation = nn.LeakyReLU()

        self.chn_fix = ChannelFixer(in_channels, self.channels, 1, 1, 0, affine=False)

        # nodes generation
        start_nodes, end_nodes = generateNodes(cell_nodes)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            dgl.DGLGraph.__init__(self, (start_nodes, end_nodes))

        self.operations_list = torch.nn.ModuleList()
        self.edge_operations_dict = {}
        self.edge_operations = edge_operations
        for edge_idx in range(self.num_edges()):
            split_O = edge_operations[edge_idx].split("$")
            self.edge_operations_dict[edge_idx] = {}
            for i in range(0, len(split_O) - 1, 2):
                self.edge_operations_dict[edge_idx][split_O[i]] = int(split_O[i + 1])

            if edge_idx < self.num_nodes() - 1:
                stride = 2
            else:
                stride = 1

            if self.edge_operations_dict[edge_idx]["operation"] in [3]:  # if convolution
                op = nn.Sequential(
                    OPERATIONS[self.edge_operations_dict[edge_idx]["operation"]](self.channels, self.channels, self.edge_operations_dict[edge_idx].get("kernel_size", 3), stride=stride, padding=math.floor(self.edge_operations_dict[edge_idx].get("kernel_size", 3) / 2), bias=False),
                    nn.BatchNorm2d(self.channels, affine=False))
            elif self.edge_operations_dict[edge_idx]["operation"] in [1, 2]:
                op = nn.Sequential(OPERATIONS[self.edge_operations_dict[edge_idx]["operation"]](self.edge_operations_dict[edge_idx].get("kernel_size", 3), stride=stride, padding=math.floor(self.edge_operations_dict[edge_idx].get("kernel_size", 3) / 2)),
                                   nn.BatchNorm2d(self.channels, affine=False))
            elif self.edge_operations_dict[edge_idx]["operation"] in [0]:
                op = ZeroOp(stride)
            elif self.edge_operations_dict[edge_idx]["operation"] in [4]:
                op = FactorizedReduce(self.channels, self.channels, affine=False) if stride == 2 else OPERATIONS[self.edge_operations_dict[edge_idx]["operation"]]()
            else:
                raise ()
            self.operations_list.append(op)

    def forward(self, batched_images):
        end_nodes = self.edges()[1]
        feature_states = {}
        feature_states[0] = self.chn_fix(batched_images)
        for node_id in range(1, self.num_nodes()):
            edge_indices = np.where(end_nodes == node_id)[0]
            updated_feats = sum(self.operations_list[edge_idx](feature_states[self.edges()[0][edge_idx].item()]) for edge_idx in edge_indices)
            feature_states[node_id] = updated_feats
        edge_indices = np.where(end_nodes == self.num_nodes() - 1)[0]
        return torch.cat([self.operations_list[edge_idx](feature_states[self.edges()[0][edge_idx].item()]) for edge_idx in edge_indices], dim=1)

    def to(self, device):
        nn.Module.to(self, device=device)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def draw(self, path, filename):
        h = gv.Digraph(edge_attr=dict(fontsize='20', ),
                       node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='1', width='1', penwidth='2'),
                       engine="dot")
        # h.graph_attr['splines'] = 'ortho'
        h.graph_attr['ranksep'] = '1 equally'
        h.graph_attr['rankdir'] = 'TB'

        for idx in self.nodes():
            if idx == 0:
                h.node(str(idx.item()), label=str(idx.item()), fillcolor='darkseagreen2')
            elif idx == len(self.nodes())-1:
                h.node(str(idx.item()), label=str(idx.item()), fillcolor='lightblue')
            else:
                h.node(str(idx.item()), label=str(idx.item()), fillcolor='palegoldenrod')
        for idx in range(len(self.edges()[0])):
            if self.edge_operations_dict[idx]["operation"] not in [0, 4]:
                h.edge(str(self.edges()[0][idx].item()), str(self.edges()[1][idx].item()), label=f"{OPERATIONS[self.edge_operations_dict[idx]['operation']].__name__}{self.edge_operations_dict.get('kernel_size', 3)}x{self.edge_operations_dict.get('kernel_size', 3)}")
            elif self.edge_operations_dict[idx]["operation"] in [4]:
                h.edge(str(self.edges()[0][idx].item()), str(self.edges()[1][idx].item()), label=f"{OPERATIONS[self.edge_operations_dict[idx]['operation']].__name__}")

        fileformat = 'png'
        h.render(path + "/" + filename, format=fileformat, view=False, cleanup=True)
