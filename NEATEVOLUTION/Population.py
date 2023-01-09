import numpy as np
from tqdm import tqdm

from Cells import generateNodes
from DARTS.OSpace import generateOSpaceCell
from NEATEVOLUTION.Species import Species, encodeSpecies


class Population:
    def __init__(self, cell_nodes, fitness_mode, mutation_rate,  max_mutation_rate):
        self.rng = np.random.default_rng(seed=666)

        self.cell_nodes = cell_nodes
        self.fitness_mode = fitness_mode
        self.mutation_rate = mutation_rate
        self.max_mutation_rate = max_mutation_rate
        self.sub_populations = {
            "normal": {},
            "reduction": {}
        }

    def __getitem__(self, item):
        return self.sub_populations[item]

    def populate(self):
        start_nodes, _ = generateNodes(self.cell_nodes)
        edge_number = len(start_nodes)
        O_space = generateOSpaceCell()
        O_space.remove("operation$0$")
        for key in self.sub_populations:
            for species_idx in range(edge_number // 2):
                for i in range(3):
                    cell_operation_list = []
                    for edge_idx in range(edge_number):
                        cell_operation_list.append(self.rng.choice(O_space, replace=True))

                    for edge_idx in self.rng.choice(range(0, edge_number), species_idx, replace=False):
                        cell_operation_list[edge_idx] = "operation$0$"
                    trait = encodeSpecies(cell_operation_list)
                    if f"Species[{trait}]" not in self.sub_populations[key].keys():
                        species = Species(f"Species[{trait}]", trait, self.fitness_mode, self.mutation_rate, self.max_mutation_rate)
                        self.sub_populations[key][species.name] = species
                    else:
                        species = self.sub_populations[key][f"Species[{trait}]"]
                    species.addIndividual(cell_operation_list)


    def reproduce(self):
        sorted_pop = self.sortSpecies()
        for sub_pop in self.sub_populations:
            sub_sorted = sorted_pop[sub_pop]
            for idx, species in enumerate(sub_sorted):
                other_species_offsprings = self.sub_populations[sub_pop][species].reproduce((4 - idx) if idx < 4 else 1)
                for other_offspring in other_species_offsprings:
                    if f"Species[{encodeSpecies(other_offspring)}]" not in self.sub_populations[sub_pop].keys():
                        self.sub_populations[sub_pop][f"Species[{encodeSpecies(other_offspring)}]"] = Species(f"Species[{encodeSpecies(other_offspring)}]", encodeSpecies(other_offspring), self.fitness_mode, self.mutation_rate, self.max_mutation_rate)
                        tqdm.write(f"New Species Discovered!! {encodeSpecies(other_offspring)}")
                    self.sub_populations[sub_pop][f"Species[{encodeSpecies(other_offspring)}]"].addIndividual(other_offspring)
        return

    def kill(self):
        for sub_pop in self.sub_populations.values():
            for species in sub_pop.values():
                species.kill()

    def sortSpecies(self):
        if self.fitness_mode == "acc":
            reverse = True
        elif self.fitness_mode == "loss":
            reverse = False
        else:
            raise ()
        sorted_species = {}
        for key in self.sub_populations:
            sorted_species[key] = [pair[0] for pair in sorted(self.sub_populations[key].items(), key=lambda item: item[1].fitness(), reverse=reverse)]
        return sorted_species

    def topKAlives(self, k):
        topK_species = {}
        for key in self.sub_populations:
            sorted_species = sorted(self.sub_populations[key], key=lambda item: item[1].alives()[1], reverse=True)[:k]
            species_list = []
            for species_name, species in sorted_species:
                species_list.append((species, species.alives()[1], species.fitness()))
            topK_species[key] = species_list
        return topK_species

    def topKFitness(self, k):
        topK_species = {}
        sorted_species = self.sortSpecies()
        for key in self.sub_populations:
            species_list = []
            for species_name in sorted_species[key][:k]:
                species = self.sub_populations[key][species_name]
                species_list.append((species, species.alives()[1], species.fitness()))
            topK_species[key] = species_list
        return topK_species

    def alives(self):
        alives = {}
        for key in self.sub_populations:
            alives[key] = []
            for spec_key, spec in self.sub_populations[key].items():
                if spec.alives()[1] > 0:
                    alives[key].append(spec_key)
        return alives