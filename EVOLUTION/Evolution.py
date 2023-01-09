import gc
import os
import re
import shutil
from itertools import combinations

from matplotlib import pyplot as plt
from tqdm import tqdm

from CIFAR10Dataset import CIFAR10Dataset
from CellNetwork import *
from Cells import EDGE_PARAMETERS
from DARTS.OSpace import generateOSpaceCell


class Evolution:
    def __init__(self, cells_num, cell_nodes, image_dim, in_channels, first_cell_channels, population, mutation_rate=0.05):
        self.rng = np.random.default_rng(seed=666)
        self.population = population
        self.networks_path = []
        self.network_ID = 0
        self.mutation_rate = mutation_rate
        self.networks_save_dir = f"EVOLUTIONNetworks"

        ###  Networks parameters  ####
        self.image_dim = image_dim
        self.in_channels = in_channels
        self.first_cell_channels = first_cell_channels

        self.parents_performances = []
        self.networks_registry = {}

        for elem in os.listdir(self.networks_save_dir):
            os.remove(f"{self.networks_save_dir}/{elem}")

        O_space = generateOSpaceCell()

        for net_id in range(population):
            cells_list = []
            reduction_position = [math.floor(cells_num * 1 / 3), math.floor(cells_num * 2 / 3)]
            edge_num = sum(range(cell_nodes - 1, 0, -1))

            for cell_idx in range(cells_num):
                cell_operation_list = []
                for edge_idx in range(edge_num):
                    cell_operation_list.append(self.rng.choice(O_space, replace=True))
                cells_list.append(("normal", cell_operation_list))

                if cell_idx in reduction_position:
                    cell_operation_list = []
                    for edge_idx in range(edge_num):
                        cell_operation_list.append(self.rng.choice(O_space, replace=True))
                    cells_list.append(("reduction", cell_operation_list))

            self.giveLifeNewNetwork(cells_list)

    def giveLifeNewNetwork(self, cells_list):
        net = CellNetwork()
        net.initialization(cells_list, self.image_dim, self.in_channels, self.first_cell_channels)
        filename = f"network{self.network_ID}.pkl"
        net.save_as_pkl(self.networks_save_dir, filename)
        self.networks_path.append(f"{self.networks_save_dir}/{filename}")
        self.network_ID += 1

        operations = ""
        for s in np.hstack([t[1] for t in cells_list]):
            operations += re.sub('\D', '', s)

        self.networks_registry[filename.removesuffix(".pkl")] = {"operations": operations}

    def killWeak(self, net_performances, to_kill, decision_mode="acc"):

        filenames = [s.split("/")[1].removesuffix(".pkl") for s in self.networks_path]
        for idx, filename in enumerate(filenames):
            self.networks_registry[filename]["statistics"] = net_performances[idx]

        if decision_mode == "acc":
            tmp_performances = [performance[0] for performance in net_performances]
            reverse = True
        elif decision_mode == "loss":
            reverse = False
            tmp_performances = [performance[1] for performance in net_performances]
        else:
            raise ()

        self.networks_path = [x for _, x in sorted(zip(tmp_performances, self.networks_path), reverse=reverse)]
        self.parents_performances = [x for _, x in sorted(zip(tmp_performances, net_performances), reverse=reverse)][:math.ceil((1 - to_kill) * self.population)]

        for _ in range(math.floor(self.population * to_kill)):
            if os.path.exists(self.networks_path[-1]):
                tqdm.write(f"{self.networks_path[-1].split('/')[1].replace('.pkl', '')} is DEAD :(")
                os.remove(self.networks_path[-1])
            else:
                tqdm.write("The file does not exist")
            self.networks_path.pop()

    def reproducePopulation(self):
        couples = [i for i in combinations(self.networks_path, 2)]
        population_diff = self.population - len(self.networks_path)
        cripples = self.rng.choice(range(population_diff), int(population_diff * 0.50), replace=False)

        for i in range(population_diff):
            couple = couples[i % len(couples)]

            gc.disable()
            with open(couple[0], "rb") as mother_file:
                mother_net = pkl.load(mother_file)
            with open(couple[1], "rb") as father_file:
                father_net = pkl.load(father_file)

            couple = [p.split("/")[1].replace(".pkl", "") for p in couple]
            gc.enable()

            mutation_flag = i in cripples

            cells_list = []

            for cell_idx in range(len(mother_net.operationList[:-1])):
                child_operation_list = []
                for edge_idx in mother_net.operationList[cell_idx].edge_operations_dict:
                    parent = self.rng.choice([mother_net, father_net])
                    edge_dict = parent.operationList[cell_idx].edge_operations_dict[edge_idx].copy()
                    op_string = ""
                    for key, value in edge_dict.items():
                        # Mutation
                        if self.rng.random() < self.mutation_rate and mutation_flag:
                            possible_choices = list(EDGE_PARAMETERS[key])
                            possible_choices.remove(value)
                            value = self.rng.choice(possible_choices)
                            tqdm.write(f"MUTATION in network{self.network_ID + 1} of {key} from parents {couple}")

                        op_string += f"{key}${value}$"

                    child_operation_list.append(op_string)
                if isinstance(mother_net.operationList[cell_idx], NormalCell):
                    cells_list.append(("normal", child_operation_list))
                else:
                    cells_list.append(("reduction", child_operation_list))

            self.giveLifeNewNetwork(cells_list)

    def evolveNetworkPopulation(self, generations, to_kill=0.4, evolve_name="EvolutionStatistics", decision_mode="acc", load_state=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_dataset = CIFAR10Dataset(path="../CIFAR-10", train=True, device="cpu", quantity=50)
        train_dataset, val_dataset = train_dataset.split(80)

        best_accs = []
        best_losses = []

        start_time = time()
        generations_times = []

        for generation in tqdm(range(generations), position=0, leave=False, desc=f"Training: ", colour="white", ncols=80):
            tqdm.write(f"\n\nGeneration {generation} started!")
            performances = self.trainNetworkPopulation(train_dataset, val_dataset, device, epochs=8)

            generations_times.append((time() - start_time) / 3600)  # tempi in ore

            acc_performances = [tup[0] for tup in performances]
            loss_performances = [tup[1] for tup in performances]

            if decision_mode == "acc":
                best_idx = np.argmax(acc_performances)
                best_losses.append(loss_performances[best_idx])
                best_accs.append(acc_performances[best_idx])
                best_nets = np.array(self.networks_path)[np.argsort(acc_performances)[::-1]]

            else:
                best_idx = np.argmin(loss_performances)
                best_losses.append(loss_performances[best_idx])
                best_accs.append(acc_performances[best_idx])
                best_nets = np.array(self.networks_path)[np.argsort(loss_performances)]

            os.makedirs("EVOLUTIONStatistics", exist_ok=True)

            plt.plot([i for i in range(generation + 1)], best_losses, 'blue', label="Generation Loss")
            plt.plot([i for i in range(generation + 1)], np.array(best_accs) / 100, 'blueviolet', label="Generation Accuracy")
            plt.title(f"Best Loss {round(best_losses[-1], 2)} and acc. {best_accs[-1]} with {best_nets[0].split('/')[1].replace('.pkl', '')}")
            plt.xlabel('Generation')
            plt.ylabel("Statistics")
            plt.legend()
            plt.savefig(f"EVOLUTIONStatistics/{evolve_name}")
            plt.close()

            plt.plot([i for i in range(generation + 1)], [generations_times[0]] + [generations_times[i] - generations_times[i - 1] for i in range(1, generation + 1)], 'green', label="Generation Time")
            plt.title(f"Generation Training Time")
            plt.xlabel('Generation')
            plt.ylabel("Time (Hours)")
            plt.legend()
            plt.savefig(f"EVOLUTIONStatistics/{evolve_name}_GenerationTime")
            plt.close()

            plt.plot(generations_times, best_losses, 'blue', label="Generation Loss")
            plt.plot(generations_times, np.array(best_accs) / 100, 'blueviolet', label="Generation Accuracy")
            plt.title(f"Best Loss {round(best_losses[-1], 2)} and acc. {best_accs[-1]} with {best_nets[0].split('/')[1].replace('.pkl', '')}")
            plt.xlabel('Elapsed Time (Hours)')
            plt.ylabel("Statistics")
            plt.legend()
            plt.savefig(f"EVOLUTIONStatistics/{evolve_name}_OverallTime")
            plt.close()

            os.makedirs("EVOLUTIONBestNetworks", exist_ok=True)
            for idx, top_net_path in enumerate(best_nets[:2]):
                shutil.copy(top_net_path, f"EVOLUTIONBestNetworks/Top{idx}Net.pkl")

            save_dict = {
                "generations_times": generations_times,
                "best_accuracies": best_accs,
                "best_losses": best_losses
            }

            with open(f"EVOLUTIONStatistics/{evolve_name}_Statistics.pkl", "wb") as stat_file:
                pkl.dump(save_dict, stat_file)

            self.killWeak(performances, to_kill, decision_mode)
            self.reproducePopulation()

    def alreadyTrained(self, filename):
        for key, value in self.networks_registry.items():
            if filename != key and self.networks_registry[filename]["operations"] == value["operations"]:
                return value["statistics"]
        return None

    def trainNetworkPopulation(self, train_dataset, test_dataset, device, epochs=8):
        net_performances = list(self.parents_performances)

        for path in self.networks_path[len(self.parents_performances):]:
            already_trained = self.alreadyTrained(path.split("/")[1].removesuffix(".pkl")) if len(self.parents_performances) > 0 else None
            if already_trained is not None:
                net_performances.append(already_trained)
                print("Network Training Skipped!")
                continue
            tqdm.write(f"\nTraining net {path.split('/')[1].replace('.pkl', '')}")
            with open(path, "rb") as net_file:
                net = pkl.load(net_file)
            net.to(device)
            net_performances.append(self.trainCellNetwork(net, train_dataset, test_dataset, epochs, device))
            tqdm.write(f"Accuracy : {net_performances[-1][0]}\n"
                       f"Loss.: {net_performances[-1][1]}\n")
            del net
            torch.cuda.empty_cache()
        return net_performances

    def trainCellNetwork(self, model, train_dataset, test_dataset, epochs, device):
        best_acc = 0
        best_loss = 1000
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters())
        BATCH_SIZE = 64
        torch.manual_seed(666)
        for epoch in tqdm(range(epochs), position=0, leave=False, desc=f"Training: ", colour="white", ncols=80):
            train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            _, _, train_loss = model.train_step(train_dataloader, optimizer, criterion, epoch, device, verbose=0)

            classification_acc, _, classification_loss, _ = model.test(test_dataset=test_dataset, criterion=criterion, batch_size=BATCH_SIZE, device=device)

            if classification_acc > best_acc:
                best_acc = classification_acc

            if classification_loss < best_loss:
                best_loss = classification_loss

        return best_acc, best_loss
