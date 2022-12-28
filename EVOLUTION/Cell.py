import math
import warnings

import dgl
import numpy as np


from Operations import *


def generate_nodes(num_nodes):
    start_nodes = np.array([i for i in range(num_nodes) for _ in range(i + 1, num_nodes)])
    end_nodes = np.array([j for i in range(num_nodes) for j in range(i + 1, num_nodes)])

    return start_nodes, end_nodes


class NoPathFound(Exception):
    pass


class Cell(nn.Module, dgl.DGLGraph):
    def __init__(self, cell_id, input_dim, in_channels, cell_nodes, cell_channels, edge_feats):
        super(Cell, self).__init__()


        self.cell_id = cell_id
        self.image_dim = input_dim
        self.channels = cell_channels

        # nodes generation
        start_nodes, end_nodes = generate_nodes(cell_nodes)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            dgl.DGLGraph.__init__(self, (start_nodes, end_nodes))

        for key, value in edge_feats.items():
            self.edata[key] = torch.tensor(value, dtype=torch.int)

        self.chn_fix_conv = nn.Conv2d(in_channels, self.channels, kernel_size=1, stride=1, dilation=1, bias=False)
        self.activation = nn.LeakyReLU()

        self.operations_list = torch.nn.ModuleList()
        self.generate_module_list()

    def generate_module_list(self):

        if len(self.operations_list) == 0:
            self.operations_list = torch.nn.ModuleList()

        for edge_idx, start_node in enumerate(self.edges()[0]):
            operation = self.edata["operation"][edge_idx].item()
            if operation == 0:  # Zero operation
                self.operations_list.append(ZeroOp())
                continue
            elif operation == 1:  # MaxPool
                self.operations_list.append(OPERATIONS[operation](self.edata["kernel_size"][edge_idx].item(), stride=1, padding=math.floor(self.edata["kernel_size"][edge_idx].item() / 2)))
            elif operation == 2:  # AvgPool
                self.operations_list.append(OPERATIONS[operation](self.edata["kernel_size"][edge_idx].item(), stride=1, padding=math.floor(self.edata["kernel_size"][edge_idx].item() / 2)))
            elif operation == 3:  # Conv
                self.operations_list.append(nn.Sequential(OPERATIONS[operation](self.channels, self.edata["channels"][edge_idx].item(), self.edata["kernel_size"][edge_idx].item(), stride=1, padding=math.floor(self.edata["kernel_size"][edge_idx].item() / 2), bias=False),
                                                          nn.BatchNorm2d(self.edata["channels"][edge_idx].item(), affine=False),
                                                          self.activation,
                                                          nn.Conv2d(self.edata["channels"][edge_idx].item(), self.channels, kernel_size=1, stride=1, dilation=1, bias=False),
                                                          nn.BatchNorm2d(self.channels, affine=False)))
            elif operation == 4:  # Identity
                self.operations_list.append(OPERATIONS[operation]())

        for node_id in range(1, self.num_nodes()):
            end_nodes = self.edges()[1]
            edge_indices = np.where(end_nodes == node_id)[0]
            # dimensions = []
            dimensions = 0
            for edge_idx in edge_indices:
                # start_node = self.edges()[0][edge_idx]
                if self.edata["operation"][edge_idx] in [1]:
                    # dimensions.append(np.floor(((self.ndata["image_dim"][start_node] + (2 * self.operations_list[edge_idx].padding) - (self.operations_list[edge_idx].dilation * (self.operations_list[edge_idx].kernel_size - 1)) - 1) / self.operations_list[edge_idx].stride) + 1))
                    dimensions += 1
                elif self.edata["operation"][edge_idx] in [2]:  # AvgPool
                    # dimensions.append(np.floor(((self.ndata["image_dim"][start_node] + (2 * self.operations_list[edge_idx].padding) - (1 * (self.operations_list[edge_idx].kernel_size - 1)) - 1) / self.operations_list[edge_idx].stride) + 1))
                    dimensions += 1
                elif self.edata["operation"][edge_idx] in [3]:  # Conv
                    dimensions += 1
                    # dimensions.append(np.floor(((self.ndata["image_dim"][start_node] + (2 * self.operations_list[edge_idx].padding[0]) - (self.operations_list[edge_idx].dilation[0] * (self.operations_list[edge_idx].kernel_size[0] - 1)) - 1) / self.operations_list[edge_idx].stride[0]) + 1))
                elif self.edata["operation"][edge_idx] in [4]:  # Identity
                    dimensions += 1
                    # dimensions.append(self.ndata["image_dim"][start_node])
                else:  # ZeroOP
                    pass
            if dimensions <= 0:  # len(dimensions) <= 0:
                raise NoPathFound("No path found!")
            # self.ndata["image_dim"][node_id] = math.ceil((dimensions[np.argmin(dimensions)] + dimensions[np.argmax(dimensions)]) / 2)

    def forward(self, batched_images):
        feature_states = {}
        feature_states[0] = self.chn_fix_conv(batched_images)
        for node_id in range(1, self.num_nodes()):
            end_nodes = self.edges()[1]
            edge_indices = np.where(end_nodes == node_id)[0]
            updated_feats = torch.zeros((batched_images.shape[0], self.channels, self.image_dim, self.image_dim), device=feature_states[0].device)
            for edge_idx in edge_indices:
                start_node = self.edges()[0][edge_idx].item()
                updated_feats += self.activation(self.operations_list[edge_idx](feature_states[start_node]))
            feature_states[node_id] = updated_feats
        return feature_states[self.num_nodes() - 1]

    def to(self, device):
        nn.Module.to(self, device=device)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
