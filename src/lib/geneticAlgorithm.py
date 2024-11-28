import random
import copy
from matplotlib import pyplot as plt

import lib.config as Config
import lib.graph as Graph
import lib.constants as Constants

class GeneticAlgorithm:
    def __init__(
            self, target, use_seed,
            budget=10, population_size=10, selection_size=3,
            crossover_rate=0.5, mutation_rate=0.5,
            mute_op=0.5, mute_type=0.5,
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
        self.mute_op = mute_op
        self.mute_type = mute_type

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
        6. push children to new population until population size is reached
        7. select fit graphs in old population + new population 
        8. repeat 3-8 until budget is reached
        """
        population = []
        
        if self.use_seed:
            print(f"[*] Using initial seeds for {self.target} dataset\n")
            initial_seed_dir = self.target_dataset_dir / "initial_seeds"
            for graph_file in initial_seed_dir.iterdir():
                # if graph_file.name != "newgraph_Facebook_equal_close_ENM_K_60.txt": continue
                individual = Graph.Graph(self.config)
                individual.init_graph()
                individual.read_graph_from_file_path(graph_file)

                # 2. evaluate individual before adding to population
                individual.evaluate_graph(self.target_dataset_dir)

                population.append(individual)

        # randomly initialize population until population size is reached
        if len(population) < self.population_size:
            print(f"[*] Randomly initializing population for {self.target} dataset\n")
            for i in range(self.population_size - len(population)):
                # choose random number of nodes to remove
                individual = copy.deepcopy(self.original_graph)
                individual.make_as_random_subgraph()
                
                individual.set_node_nums()
                individual.evaluate_graph(self.target_dataset_dir)

                population.append(individual)
        
        # randomly select a graph as initial best_graph
        best_graph = population[random.randrange(self.population_size)]
        print(best_graph)

        gen_count = 0
        roc_curve = []
        print(f"[*] Running GA for {self.target} dataset\n")
        while gen_count < self.budget or best_graph.fitness_score == 0.0:
            next_gen = []
            while len(next_gen) < self.population_size:
                # 5. mutate childrent
                o1 = self.select(self.selection_size, population)
                self.mutate_graph(o1, self.mutation_rate, self.mute_op, self.mute_type)
                next_gen.append(o1)
            
            # 6. extend population with next_gen
            population.extend(next_gen)
            population = sorted(population, key=lambda x: x.fitness_score, reverse=False)
            population = population[:self.population_size]

            # 7. select fit graphs in old population + new population
            best_graph = population[0]
            gen_count += 1
            roc_curve.append(best_graph.fitness_score)
            print(f"[*] Generation {gen_count}/{self.budget} best graph: {best_graph.fitness_score}")
            
        self.plot_graph(roc_curve)
        self.write_best_graph(best_graph)
        return best_graph
    
    def plot_graph(self, roc_curve):
        plt.plot(roc_curve)
        plt.ylabel('Fitness Score')
        plt.xlabel('Generation')

        # save to file
        png_file = Constants.result_dir_path / f"roc_curve_{self.target}.png"
        plt.savefig(png_file)
    
    def write_best_graph(self, best_graph):
        score_file = Constants.result_dir_path / f"best_graph_{self.target}.csv"
        with open(score_file, "w") as fp:
            fp.write("Generation,Score\n")
            for gen, score in enumerate(best_graph.fitness_score_history):
                content = f"{gen},{score}\n"
                fp.write(content)

        mutation_file = Constants.result_dir_path / f"mutation_history_{self.target}.csv"
        with open(mutation_file, "w") as fp:
            fp.write("Generation,mutation_op\n")
            for gen, mutation_op in enumerate(best_graph.mutation_history):
                content = f"{gen},{mutation_op}\n"
                fp.write(content)



    
    def select(self, k, population):
        # we randomly sample k solutions from the population
        participants = random.sample(population, k)
        # fitness_values = [fitness(p) for p in participants]
        result = sorted(participants, key=lambda x:x.fitness_score, reverse=False)[0]
        return copy.deepcopy(result)
        
    def mutate_graph(self, g1, mutation_rate=0.5, mute_op=0.5, mute_type=0.5):
        """
        returns a mutated version of the graph,
        utilizes function for add, remove, of nodes and edges
        in graph class
        """
        
        if random.random() < mutation_rate:
            if random.random() < mute_op:
                if random.random() < mute_type:
                    g1.add_random_nodes()
                else:
                    g1.add_random_edges()
            else:
                if random.random() < mute_type:
                    g1.remove_random_nodes()
                else:
                    g1.remove_random_edges()
        
        g1.set_node_nums()
        g1.evaluate_graph(self.target_dataset_dir)
        