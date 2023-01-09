import re
from itertools import combinations

import numpy as np

from DARTS.OSpace import generateOSpaceCell


class Species:
    def __init__(self, name, trait, fitness_mode, mutation_rate, max_mutation_rate):
        self.rng = np.random.default_rng(seed=11)

        self.name = name

        self.trait = trait
        self.fitness_mode = fitness_mode
        self.mutation_rate = mutation_rate
        self.max_mutation_rate = max_mutation_rate

        self.individuals = {}

        self.individuals_fitness = {}
        self.individuals_fitness_sum = {}
        self.individuals_usage = {}

        self.alive_individuals = {}

    def fitness(self):
        ind_fitness = [self.individuals_fitness[ind_key] for ind_key in self.alives()[0] if self.individuals_fitness[ind_key] > 0]
        if ind_fitness:
            return np.mean(ind_fitness)
        return 0

    def updateIndividualFitness(self, individual_op_list, fitness):
        if not self.sameSpecies(individual_op_list):
            raise ()

        key = encodeOpList(individual_op_list)
        self.individuals_fitness_sum[key] += fitness
        self.individuals_usage[key] += 1
        self.individuals_fitness[key] = self.individuals_fitness_sum[key] / self.individuals_usage[key]

    def reproduce(self, offspring_number):

        other_species = []

        if len(self.alives()[0]) == 0:
            return other_species

        if len(self.alives()[0]) == 1:
            parent_key = self.alives()[0][0]
            for i in range(offspring_number):
                offspring = self.mutate(self.individuals[parent_key])
                if self.sameSpecies(offspring):
                    self.addIndividual(offspring)
                else:
                    other_species.append(offspring)
            return other_species

        couples = [i for i in combinations(self.sortIndividuals(), 2)]

        for i in range(offspring_number):
            mother_key, father_key = couples[i % len(couples)]

            offspring_op_list = []
            for edge_idx in range(len(self.individuals[mother_key])):
                parent_key = self.rng.choice([mother_key, father_key])
                offspring_op_list.append(self.individuals[parent_key][edge_idx])

            if self.rng.random() < self.mutation_rate:
                offspring_op_list = self.mutate(offspring_op_list)

            if self.sameSpecies(offspring_op_list):
                self.addIndividual(offspring_op_list)
            else:
                other_species.append(offspring_op_list)

        return other_species

    def mutate(self, individual_op_list):
        O_space = generateOSpaceCell()
        mutated_op_list = []
        number_of_mutations = self.rng.integers(1, int(len(individual_op_list) * self.max_mutation_rate), endpoint=True)
        mutation_pos = self.rng.choice(len(individual_op_list), number_of_mutations, replace=False)
        for op_idx, operation in enumerate(individual_op_list):
            if op_idx in mutation_pos:
                candidates = O_space.copy()
                candidates.remove(operation)
                mutated_op_list.append(self.rng.choice(candidates))
            else:
                mutated_op_list.append(operation)
        return mutated_op_list

    def kill(self):
        kill_thr = 2
        sorted_individuals = self.sortIndividuals()
        for idx in range(len(sorted_individuals) - 1, -1, -1):
            ind_key = sorted_individuals[idx]
            if kill_thr <= 0:
                break
            if kill_thr < self.alive_individuals[ind_key]:
                self.alive_individuals[ind_key] -= kill_thr
                kill_thr = 0
            else:
                kill_thr -= self.alive_individuals[ind_key]
                self.alive_individuals[ind_key] = 0
                self.individuals_fitness[ind_key] = 0
                self.individuals_fitness_sum[ind_key] = 0
                self.individuals_usage[ind_key] = 0

    def sortIndividuals(self):
        if self.fitness_mode == "acc":
            reverse = True
        elif self.fitness_mode == "loss":
            reverse = False
        else:
            raise ()
        sorted_pairs = sorted(self.individuals_fitness.items(), key=lambda item: item[1], reverse=reverse)
        return [pair[0] for pair in sorted_pairs]

    def topKAlives(self, k):
        sorted_individuals = sorted(self.alive_individuals.items(), key=lambda item: item[1], reverse=True)[:k]
        ind_list = []
        for ind_key, number in sorted_individuals:
            ind_list.append((self.individuals[ind_key], number, self.individuals_fitness[ind_key]))
        return ind_list

    def topKFitness(self, k):
        sorted_individuals = self.sortIndividuals()[:k]
        ind_list = []
        for ind_key in sorted_individuals:
            ind_list.append((ind_key, self.alive_individuals[ind_key], self.individuals_fitness[ind_key]))
        return ind_list

    def alives(self):
        alive_values = list(self.alive_individuals.values())
        alives_number = sum(alive_values)
        alives_mask = np.array(alive_values) > 0
        alive_keys = np.array(list(self.alive_individuals.keys()))[alives_mask]
        return alive_keys, alives_number

    def sameSpecies(self, individual_op_list):
        return self.trait == encodeSpecies(individual_op_list)

    def addIndividual(self, individual_op_list):
        if not self.sameSpecies(individual_op_list):
            return
        offspring_key = encodeOpList(individual_op_list)
        self.individuals[offspring_key] = individual_op_list
        if self.alive_individuals.get(offspring_key, -1) == -1:
            self.alive_individuals[offspring_key] = 1
            self.individuals_fitness[offspring_key] = 0
            self.individuals_fitness_sum[offspring_key] = 0
            self.individuals_usage[offspring_key] = 0
        else:
            self.alive_individuals[offspring_key] += 1


def encodeOpList(op_list):
    return re.sub('\D', '', str(op_list))


def encodeSpecies(op_list):
    zero_idx = np.where(np.array(op_list) == "operation$0$")[0]
    return len(zero_idx)
    # encode = ""
    # for el in zero_idx:
    #     encode += str(el)
    # return encode
