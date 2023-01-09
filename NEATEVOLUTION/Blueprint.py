import math


class Blueprint(list):
    def __init__(self, cells_num):
        cells_list = []
        reduction_position = [math.floor(cells_num * 1 / 3), math.floor(cells_num * 2 / 3)]

        for cell_idx in range(cells_num):
            cells_list.append("normal")
            if cell_idx in reduction_position:
                cells_list.append("reduction")
        super().__init__(cells_list)
