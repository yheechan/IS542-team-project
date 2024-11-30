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
        args.experiment_name, args.target, args.use_seed,
        budget=500, population_size=40, selection_size=3,
        crossover_rate=0.5, mutation_rate=0.8,
        mute_op=0.5, mute_type=0.5,
    )
    ga.run_ga()

    


if __name__ == "__main__":
    parser = ArgParser.ArgParser()
    args = parser.parse_args()
    main(args)

    