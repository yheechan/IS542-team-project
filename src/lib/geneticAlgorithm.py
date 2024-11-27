import random

import lib.config as Config
import lib.graph as Graph
import lib.constants as Constants

class GeneticAlgorithm:
    def __init__(
            self, target, use_seed,
            budget=10, population_size=10, selection_size=3,
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
        self.selection_size = selection_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        # intializes original graph
        self.original_graph = Graph.Graph(self.config)
        self.original_graph.init_graph()
        
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

        # 1. if use_seed option is True, then load initial seed (graphs)
        if self.use_seed:
            initial_seed_dir = self.target_dataset_dir / "initial_seeds"
            for graph_file in initial_seed_dir.iterdir():
                individual = Graph.Graph(self.config)
                individual.init_graph()

                individual.read_graph_from_file_path(graph_file)
                # 2. evaluate individual before adding to population
                individual.prior_file_path = self.target_dataset_dir / "train.txt"
                individual.withBuffer = True
                individual.target_file_path = self.target_dataset_dir / "target_close.txt"
                individual.evaluate_graph()

                population.append(individual)

        # randomly initialize population until population size is reached
        for i in range(self.population_size - len(population)):
            individual = self.original_graph#.random_subgraph()
        
        # randomly select a graph as initial best_graph
        best_graph = population[random.randrange(len(population))]
        print(f"Initial best graph: {best_graph}")

        gen_count = 0
        while gen_count < self.budget or best_graph.fitness_score == 0.0:
            next_gen = []

            while len(next_gen) < len(population):
                # 3. randomly select parents
                p1 = self.select(self.selection_size, population)
                p2 = self.select(self.selection_size, population)

                # 4. crossover parents
                o1, o2 = self.crossover_graph(p1, p2, crossover_rate=self.crossover_rate)
                break
            break
    
    def select(self, k, population):
        # we randomly sample k solutions from the population
        participants = random.sample(population, k)
        # fitness_values = [fitness(p) for p in participants]
        result = sorted(participants, key=lambda x:x.fitness_score, reverse=False)
        return result[0]

        

        


        
        
    
    def crossover_graph(self, g1, g2, crossover_rate=0.5):
        """
        returns two graph that results from crossover
        """
        if random.random() < crossover_rate:
            pass

        return g1, g2

    def mutate_graph(self, mutation_rate, g1):
        """
        returns a mutated version of the graph,
        utilizes function for add, remove, of nodes and edges
        in graph class
        """
        # TODO: implement this function
        pass
    