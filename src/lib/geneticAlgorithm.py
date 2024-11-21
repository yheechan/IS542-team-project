
import lib.config as Config
import lib.graph as Graph
import lib.constants as Constants

class GeneticAlgorithm:
    def __init__(
            self, target, use_seed,
            budget=10, population_size=10,
            crossover_rate=0.5, mutation_rate=0.5
    ):
        # initializes configs
        self.target = target
        self.use_seed = use_seed
        self.target_dataset_dir = Constants.dataset_dir_path / target
        self.config = Config.Config(target)

        # initializes GA parameters
        self.budget = budget
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        # intializes original graph
        self.original_graph = Graph.Graph(self.config)
        # self.original_graph.init_graph(self.config)
        
        # # dataset/Facebook/originalgraph_Facebook.txt
        self.original_graph.read_graph_from_file_path(
            f"{self.target_dataset_dir}/originalgraph_{self.target}.txt"
        )
    
    def run_ga(self):
        """
        runs genetic algorithm
        1. intializes random population
        2. evaluates population
        3. randomly select parents
        4. crossover parents
        5. mutate children
        6. evaluate children
        7. push children to new population until population size is reached
        8. select fit graphs in old population + new population 
        9. repeat 3-8 until budget is reached
        """
        # TODO: implement this function
        population = []

        # if use_seed option is True, then load initial seed (graphs)
        if self.use_seed:
            initial_seed_dir = self.target_dataset_dir / "initial_seeds"
            for graph_file in initial_seed_dir.iterdir():
                individual = Graph.Graph(self.config)
                individual.read_graph_from_file_path(graph_file)
                print(individual.node_num)
                population.append(individual)

        # # randomly initialize population until population size is reached
        # for i in range(self.population_size - len(population)):
        #     individual = self.original_graph#.random_subgraph()
        
        # # assign fitness for each individual in the population
        # for i in range(len(population)):
        #     population[i].evaluate_graph()
            
        


        
        
    
    def crossover_graph(self, crossover_rate, g1, g2):
        """
        returns two graph that results from crossover
        """
        # TODO: implement this function
        pass

    def mutate_graph(self, mutation_rate, g1):
        """
        returns a mutated version of the graph,
        utilizes function for add, remove, of nodes and edges
        in graph class
        """
        # TODO: implement this function
        pass
    