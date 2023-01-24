import math
import os
import datetime
import torch
from torch import nn
from tqdm import tqdm
import pickle as pkl
from CIFAR10Dataset import CIFAR10Dataset
from CellNetwork import CellNetwork
from Cells import NormalCell
from NEATEVOLUTION.Blueprint import Blueprint
from NEATEVOLUTION.Population import Population
import numpy as np

from NEATEVOLUTION.Species import encodeSpecies
from PlotUtils import plotTopSpecies


class NEATEvolution:
    def __init__(self, cells_num, cell_nodes):

        # INPUT PARAMETERS
        self.IMAGE_INPUT_DIM = 32
        self.IN_CHANNELS = 3
        self.FIRST_CELL_CHN = 16

        # TRAINING PARAMETERS
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW
        self.BATCH_SIZE = 256
        self.EPOCHS = 8
        self.NETWORKS_NUMBER = 20
        self.DATASET_SIZE = 30

        # EVOLVE PARAMETERS
        self.GENERATIONS = 36
        self.FITNESS_MODE = "acc"
        self.MUTATION_RATE = 0.20
        self.MAX_MUTATION_RATE = 0.30
        self.UNIFORM_SPECIES_CHOOSE = True

        # PATHS
        self.save_path = "Populations"
        self.data = datetime.datetime.now().strftime("%d%m%Y_%H%M")

        self.rng = np.random.default_rng(137)

        self.population = Population(cell_nodes, self.FITNESS_MODE, self.MUTATION_RATE, self.MAX_MUTATION_RATE)
        self.population.populate()

        self.blueprints = [Blueprint(cells_num)]

    def assembleNetwork(self):
        blueprint = self.rng.choice(self.blueprints)
        network_cells = []

        alives = self.population.alives()

        species_for_cell = {}
        sub_pop_modules = {}
        for cell_type in np.unique(blueprint):
            if self.UNIFORM_SPECIES_CHOOSE:
                probabilities = None
            else:
                probabilities = exp_distr(len(list(alives[cell_type])), lam=0.15)
            species_name = self.rng.choice(alives[cell_type], p=probabilities)
            species_for_cell[cell_type] = self.population[cell_type][species_name]

            sub_pop_modules[cell_type] = []

            sorted_individuals = self.population[cell_type][species_name].sortIndividuals()
            sorted_alive_individuals = [ind for ind in sorted_individuals if ind in self.population[cell_type][species_name].alives()[0]]

            # Choose with 100% of probability the newborns in the species
            count = len(sorted_alive_individuals) - 1
            while self.population[cell_type][species_name].individuals_fitness[sorted_alive_individuals[count]] == 0 and len(sub_pop_modules[cell_type]) < len(blueprint) and count >= 0 :
                sub_pop_modules[cell_type].append(sorted_alive_individuals[count])
                count -= 1

            # Create the list of candidates modules considering all the alive individuals
            modules_candidates = []
            for key, alive_num in self.population[cell_type][species_name].alive_individuals.items():
                for _ in range(alive_num):
                    modules_candidates.append(key)

            # Choose a module from candidates for the remaining cells
            while len(sub_pop_modules[cell_type]) < len(blueprint):
                sub_pop_modules[cell_type].append(self.rng.choice(modules_candidates, replace=True))

            # Shuffle the modules
            self.rng.shuffle(sub_pop_modules[cell_type])

        for cell_idx, cell_type in enumerate(blueprint):
            species = species_for_cell[cell_type]

            module_cell = sub_pop_modules[cell_type][cell_idx]

            network_cells.append((cell_type, self.population[cell_type][species.name].individuals[module_cell]))

        network = CellNetwork()
        network.initialization(network_cells, self.IMAGE_INPUT_DIM, self.IN_CHANNELS, self.FIRST_CELL_CHN)
        return network

    def trainNetwork(self, network, train_dataset, test_dataset):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        network.to(device)

        best_acc = 0
        best_loss = 1000
        optimizer = self.optimizer(network.parameters())
        torch.manual_seed(666)
        for epoch in tqdm(range(self.EPOCHS), position=2, leave=False, desc=f"Epochs: ", colour="white", ncols=60):
            train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
            _, _, train_loss = network.train_step(train_dataloader, optimizer, self.criterion, epoch, device, verbose=0)

            classification_acc, _, classification_loss, _ = network.test(test_dataset=test_dataset, criterion=self.criterion, batch_size=self.BATCH_SIZE, device=device)

            if classification_acc > best_acc:
                best_acc = classification_acc

            if classification_loss < best_loss:
                best_loss = classification_loss

        return best_acc, best_loss

    def generation(self, train_dataset, test_dataset):

        used_cells = []
        for net_id in tqdm(range(self.NETWORKS_NUMBER), position=1, leave=False, desc=f"Training Networks: ", colour="white", ncols=80):
            network = self.assembleNetwork()
            performance = self.trainNetwork(network, train_dataset, test_dataset)[0 if self.FITNESS_MODE == "acc" else 1]

            for cell in network.operationList[:-1]:
                species = encodeSpecies(cell.edge_operations)
                sub_pop = "normal" if isinstance(cell, NormalCell) else "reduction"
                used_cells.append((sub_pop, species, cell.edge_operations, performance))

        for sub_pop, species, edge_operations, performance in used_cells:
            self.population[sub_pop][f"Species[{species}]"].updateIndividualFitness(edge_operations, performance)

    def evolve(self):
        train_dataset = CIFAR10Dataset(path="../CIFAR-10", train=True, device="cpu", quantity=self.DATASET_SIZE)
        train_dataset, val_dataset = train_dataset.split(80)

        save_path = "NEATEvolutionResults"
        os.makedirs(save_path, exist_ok=True)

        for generation in tqdm(range(self.GENERATIONS), position=0, leave=False, desc=f"Generation: ", colour="white", ncols=100):

            if generation > 0:
                self.UNIFORM_SPECIES_CHOOSE = True

            self.generation(train_dataset, val_dataset)

            if generation > 2:
                plotTopSpecies(self.population, save_path, show=True)

            self.savePopulation(self.save_path)

            # tqdm.write("########  PRE-KILLING   ###########")
            # top_species = self.population.topKFitness(3)["reduction"]
            # tqdm.write(f"{top_species}")
            # top_individuals = self.population["reduction"][top_species[0][0].name].topKFitness(3)
            # tqdm.write(f"{top_individuals}")

            self.population.kill()

            # tqdm.write("########  POST-KILLING & PRE-REPRODUCE  ###########")
            # top_species = self.population.topKFitness(3)["reduction"]
            # tqdm.write(f"{top_species}")
            # top_individuals = self.population["reduction"][top_species[0][0].name].topKFitness(3)
            # tqdm.write(f"{top_individuals}")

            self.population.reproduce()
            #
            # tqdm.write("########  POST-REPRODUCE   ###########")
            # top_species = self.population.topKFitness(3)["reduction"]
            # tqdm.write(f"{top_species}")
            # top_individuals = self.population["reduction"][top_species[0][0].name].topKFitness(3)
            # tqdm.write(f"{top_individuals}")

    def savePopulation(self, path):
        os.makedirs(path, exist_ok=True)

        with open(f"{path}/population{self.data}.pkl", "wb") as pop_file:
            pkl.dump(self.population, pop_file)

    def loadPopulation(self, path, filename):
        with open(f"{path}/{filename}.pkl", "rb") as pop_file:
            self.population = pkl.load(pop_file)

    def bestNetwork(self, normal_layers, best_normal_species_idx=0, best_reduction_species_idx=0):

        REDUCTION_POS = [math.floor(normal_layers / 3), math.floor(normal_layers / 3 * 2)]

        best_species = self.population.topKFitness(10)

        normal_best_species = best_species["normal"][best_normal_species_idx][0]
        best_individual_key = normal_best_species.topKFitness(1)[0][0]
        best_normal_cell = ("normal", normal_best_species.individuals[best_individual_key])

        reduction_best_species = best_species["reduction"][best_reduction_species_idx][0]
        best_individual_key = reduction_best_species.topKFitness(1)[0][0]
        best_reduction_cell = ("reduction", reduction_best_species.individuals[best_individual_key])



        first_block = [best_normal_cell] * REDUCTION_POS[0]
        second_block = [best_normal_cell] * REDUCTION_POS[0]
        last_block = [best_normal_cell] * (normal_layers - (2 * REDUCTION_POS[0]))
        best_net = first_block + [best_reduction_cell] + second_block + [best_reduction_cell] + last_block

        return best_net



def exp_distr(array_length, lam):
    x = np.arange(array_length)
    p = lam * np.exp((-1 * lam * x))
    p /= sum(p)
    return p
