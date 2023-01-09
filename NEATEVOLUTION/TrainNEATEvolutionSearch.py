import os
import sys

from NEATEVOLUTION.NEATEvolution import NEATEvolution
from PlotUtils import plotTopSpecies


def main():
    os.chdir(os.path.dirname(sys.argv[0]))

    evolution = NEATEvolution(7, 5)
    # evolution.loadPopulation("Populations", "population07012023_1807")
    evolution.evolve()


if __name__ == "__main__":
    main()
