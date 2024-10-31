import lib.argParser as ArgParser
import lib.config as Config
import lib.graph as Graph
import lib.constants as Constants
import lib.geneticAlgorithm as GeneticAlgorithm

# python3 main.py 
# -t Facebook 
# -a ENM 
# -f newgraph_Facebook_equal_close_ENM.txt
# -o originalgraph_Facebook.txt

def main(args):
    # initializes GA
    ga = GeneticAlgorithm.GeneticAlgorithm(
        args.target,
        budget=10, population_size=10,
        crossover_rate=0.5, mutation_rate=0.5
    )

    


if __name__ == "__main__":
    parser = ArgParser.ArgParser()
    args = parser.parse_args()
    main(args)

    