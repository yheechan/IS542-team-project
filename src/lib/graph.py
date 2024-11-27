import lib.config as Config
import lib.constants as Constants
import copy
import random
from sklearn.metrics import roc_auc_score
from collections import defaultdict

class Graph:
    def __init__(self, config):
        self.config = config
        # intializes..
        # self.graph
        # self.init_graph()

    ###########################
    ### function of grpah #####
    ###########################
    def remove_nodes_based_on_list(self, nodes_to_remove):
        """
        returns a copy of the graph that results from removing
        nodes in 'nodes_to_remove' from the original graph
        """

        removed_nodes_idx = 0
        for i, connected_nodes in enumerate(self.graph):
            if i not in nodes_to_remove:
                # Remove edges to nodes that will be removed
                for node in connected_nodes:
                    if node in nodes_to_remove:
                        connected_nodes.remove(node)
            else:
                # Remove nodes
                # print(f"[*] Removed nodes: {nodes_to_remove[removed_nodes_idx]}/{self.graph[i]}")
                self.graph[i] = [-1]
                removed_nodes_idx += 1
                # print(f"[*] Removed all connected nodes for node {i}")
        

    def make_as_random_subgraph(self):
        """
        returns a copy of a random grpah that is
        generated as a subgraph of the original graph
        """
    
        # Decide how many nodes will be removed: num_nodes2remove
        num_nodes2remove = random.randint(1, 5)
        print(f"[*] Number of nodes to remove: {num_nodes2remove}")

        # Decide which nodes will be removed: nodes_to_remove
        # ranges from [num_negative, node_cnt)
        nodes_to_remove = sorted(
            random.sample(
                range(self.config.num_negative, self.config.node_cnt),
                num_nodes2remove
            ))
        print(f"[*] Removing nodes: {nodes_to_remove}")
          
        # Remove nodes in graph based on the list of nodes to remove
        self.remove_nodes_based_on_list(nodes_to_remove)
        
        # key: old_idx, value: new_idx
        indexing_dict = self.get_renumber_dict_remove(nodes_to_remove)
        self.apply_renumber_dict(indexing_dict, nodes_to_remove)
        
    def apply_renumber_dict(self, indexing_dict, nodes_to_remove):
        new_graph = []
        #self.graph: remove nodes, but not yet renumbered
        for i, connected_nodes in enumerate(self.graph):
            # change indexes in connected_nodes to new indexes
            if i not in nodes_to_remove:
                assert connected_nodes[0] != -1
                assert set(connected_nodes).intersection(set(nodes_to_remove)) == set()
                new_connected_nodes = [indexing_dict[node] for node in connected_nodes]
                new_graph.append(new_connected_nodes)
            else:
                print(f"[*] Node {i} and its connected nodes are not added to new_graph")
        
           
        # assert code
        '''
        new_node_index = list(range(len(new_graph)))
        # Check node_renumbering
        assert new_node_index == list(new_index.values())
        # Check edge_renumbering
        for i, connected_nodes in enumerate(self.graph):
            # i: old_idx, new_index[i]: new_idx
            if (i in new_index):
                assert len(self.graph[i]) == len(new_graph[new_index[i]]),f"len(graph[{i}]): {len(self.graph[i])}, len(new_graph[new_index[{i}]]): {len(new_graph[new_index[i]])}"
        '''
    
        self.graph = new_graph

    def get_renumber_dict_remove(self, nodes_to_remove):
        """
        Renumber the nodes after removing some nodes.
        """
        node_size = self.node_num
        new_index = {}
        new_idx = 0
        
        print(f"[*] Nodes to remove: {nodes_to_remove}")
        # Create a mapping from old indices to new indices
        gap = 0
        for old_idx in range(node_size):
            if old_idx not in nodes_to_remove:
                # key: old_idx, value: new_idx
                new_index[old_idx] = new_idx
                new_gap = old_idx - new_idx
                if new_gap > gap:
                    if old_idx - 1 in nodes_to_remove:
                        remove_node_idx = nodes_to_remove.index(old_idx - 1)
                    else:
                        raise ValueError('Node not found')
                    print(f"[*] old_idx: {old_idx}, new_idx: {new_idx}"
                          +f"/ remove_node_idx: {remove_node_idx}, old_idx - new_idx: {new_gap}")
                    gap = new_gap
                new_idx += 1
        return new_index    
        
            

    def remove_random_nodes(self):
        """
        returns a copy of the graph that results from removing
        'number_of_nodes' number of nodes from the original graph
        """
        # TODO: implement this function
        pass

    def add_random_nodes(self):
        """
        returns a copy of the graph that results from adding
        'number_of_nodes' number of nodes to the original graph
        """
        # TODO: implement this function
        pass

    def remove_random_edges(self):
        """
        returns a copy of the graph that results from removing
        'number_of_edges' number of edges from the original graph
        """
        # TODO: implement this function
        pass

    def add_random_edges(self):
        """
        returns a copy of the graph that results from adding
        'number_of_edges' number of edges to the original graph
        """
        # TODO: implement this function
        pass

    def evaluate_graph(self, dataset_dir):
        """
        evaluates graph to SybilSCAR to
        assign a fitness score of the graph
        """
        self.prior_file_path = dataset_dir / "train.txt"
        self.withBuffer = True
        self.target_file_path = dataset_dir / "target_close.txt"
        self.sybilscar(is_train=True)
        self.check_FN_nodes()

    def __str__(self):
        ret = "*** Graph Fitness Score ***\n"
        ret += "fitness: {}\n".format(self.fitness_score)
        ret += "********************\n"
        return ret


    ###########################
    ### GRAPH UTILS from RICC ###########
    ###########################
    def init_graph(self):
        self.graph = [[] * self.config.node_cnt for _ in range(self.config.node_cnt)]

    def read_graph_from_file_path(self, file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                node1 = int(line[0])
                node2 = int(line[1])
                if node1 >= len(self.graph):
                    extend_amnt = [[] for _ in range(node1 - (len(self.graph)-1))]
                    self.graph.extend(extend_amnt)
                self.graph[node1].append(node2)
        
        self.set_node_nums()
    
    def set_node_nums(self):
        self.num_new_nodes = 0
        self.num_removed_nodes = 0

        if len(self.graph) > self.config.node_cnt:
            self.num_new_nodes = len(self.graph) - self.config.node_cnt
        elif len(self.graph) < self.config.node_cnt:
            self.num_removed_nodes = self.config.node_cnt - len(self.graph)

        self.node_num = len(self.graph)
    
    # I THINK THIS IS FUNCTION THAT RUNS SYBILSCAR
    def sybilscar(self, is_train=False):
        self.train_node_lists = []
        self.prior_list = [0] * self.node_num
        self.prior_list = self.read_prior(self.prior_list, is_train=is_train)
        
        post_list = copy.deepcopy(self.prior_list)

        for _ in range(Constants.iteration):
            # print("Iteration: ", _)
            post_list_tmp = self.run_lbp(post_list, withBuffer=self.withBuffer)
            post_list = copy.deepcopy(post_list_tmp)
        
        self.post_list = post_list

    
    def read_prior(self, prior_list, is_train=False):
        if is_train:
            with open(self.prior_file_path, "r") as f:
                train_negative = f.readline()
                train_positive = f.readline()

                self.negative_nodes = train_negative.split()
                self.positive_nodes = train_positive.split()

                for negative_idx in self.negative_nodes:
                    prior_list[int(negative_idx)] = -1 * Constants.theta
                    self.train_node_lists.append(int(negative_idx))
                for positive_idx in self.positive_nodes:
                    prior_list[int(positive_idx)] = +1 * Constants.theta
                    self.train_node_lists.append(int(positive_idx))
        else:
            with open(self.prior_file_path, "r") as f:
                lines = f.read().splitlines()

                for line in lines:
                    line = line.split()

                    node_idx = int(line[0])
                    prior_score = float(line[1])

                    prior_list[node_idx] = prior_score
        
        return prior_list


    '''
    def save_posterior(self, file_path, post_list):
        """ save posterior scores in file.
        :return:
        """
        f_post = open(file_path, "w")

        for line in enumerate(post_list):
            f_post.write('{} {:.10f}\n'.format(line[0], line[1]))
            
        f_post.close()
    
    def choose_random_train_set(self):
        turn = 0

        self.random_train_negative = []
        self.random_train_positive = []

        while True:
            rand = random.choice(self.negative_nodes)
            if rand not in self.random_train_negative:
                self.random_train_negative.append(rand)
            if len(self.random_train_negative) == Constants.sampling_size:
                break
        
        while True:
            rand = random.choice(self.positive_nodes)
            if rand not in self.random_train_positive:
                self.random_train_positive.append(rand)
            if len(self.random_train_positive) == Constants.sampling_size:
                break
        
        train_path = Constants.ricc_dir_path / f"train_{turn}_prime.txt"
        with open(train_path, "w") as f_train:
            for i in range(Constants.sampling_size):
                f_train.write(f"{self.random_train_negative[i]} ")
            f_train.write("\n")

            for i in range(Constants.sampling_size):
                f_train.write(f"{self.random_train_positive[i]} ")
            f_train.write("\n")
    
    def trainset2prior(self):
        turn = 0
        num_negative = self.config.num_negative
        num_positive = self.config.num_positive
        num_unlabel = Constants.num_unlabel

        train_score_file_path = Constants.ricc_dir_path / f"prior_{turn}_prime.txt"
        with open(train_score_file_path, "w") as f_train_score:
            for i in range(0, num_negative + num_positive + num_unlabel):
                if str(i) in self.random_train_negative:
                    f_train_score.write(str(i) + " -{}\n".format(Constants.theta))
                elif str(i) in self.random_train_positive:
                    f_train_score.write(str(i) + " {}\n".format(Constants.theta))
                else:
                    f_train_score.write(str(i) + " 0\n")
    '''

    def check_FN_nodes(self):
        turn = 1
        num_negative = self.config.num_negative
        num_positive = self.config.num_positive
        num_unlabel = Constants.num_unlabel


        # read target nodes
        f_target = open(self.target_file_path, "r")
        target_detect = 0
        target_list = []
        targets = f_target.readline()
        targets = targets.split()
        target_num = len(targets)
        for target in targets:
            target_list.append(int(target))
        
        # read original trainset
        f_train_ori = open(self.prior_file_path, "r")
        train_ori = []
        lines = f_train_ori.read().splitlines()
        for line in lines:
            line = line.split()
            train_ori.extend(line)
        
        # compute the posterior score after defense in this turn
        score_list_no_train = []
        for idx, score in enumerate(self.post_list):
            if idx not in self.train_node_lists:
                score_list_no_train.append(score)

        
        # find the FN nodes
        '''
        turn = 0
        if turn == 0:
            f1 = open(Constants.ricc_dir_path / f"post_sybilscar_before_attack({Constants.weight}_{Constants.iteration}).txt", "r")
            f2 = open(Constants.ricc_dir_path / f"post_sybilscar_0_evaluation.txt", "r")

            lines1 = f1.read().splitlines()
            lines2 = f2.read().splitlines()

            for line1, line2 in zip(lines1, lines2):
                line1 = line1.split()
                line2 = line2.split()

                if float(line1[1]) > Constants.threshold > float(line2[1]):
                    if int(line1[0]) >= num_negative:
                        Constants.FN_nodes.append(line1[0])
            
            f1.close()
            f2.close()
        
        # compute the error rates (FN rate)
        error = 0
        for node in Constants.FN_nodes:
            if float(score_list[int(node)]) < Constants.threshold:
                error += 1
        '''

        for node in target_list:
            if float(self.post_list[int(node)]) > Constants.threshold:
                target_detect += 1

        print(f"[*] graph size: {self.node_num}, {len(self.graph)}")
        print(f"[*] num_new_nodes: {self.num_new_nodes}, num_removed_nodes: {self.num_removed_nodes}")
        print(f"[*] num_negative: {num_negative}, num_positive: {num_positive}")
        y_true = [0] * (num_negative - 100) \
            + [1] * (num_positive - 100 + self.num_new_nodes - self.num_removed_nodes)
        # y_true = [0] * (num_negative - 100) \
            # + [1] * (num_positive - (100 + self.num_new_nodes))
        print(f"len(y_true): {len(y_true)}, len(score_list_no_train): {len(score_list_no_train)}")
        roc_auc = roc_auc_score(y_true, score_list_no_train)

        # print and save the performance of the RICC
        msg = "FN rate : [{}/{} ({:.0f}%)]\t AUC : {:.4f}\n".format(
            target_detect, target_num, (target_detect / target_num)*100.0, roc_auc
        )
        
        self.fitness_score = roc_auc

        f_target.close()
        f_train_ori.close()

        
    
    ###########################
    ### PRINT UTILS ###########
    ###########################
    def print_graph_info(self):
        print("Graph Info")
        print("Graph Type: ", self.config.graph_type)
        print("Node Count: ", self.node_num)

    def print_graph(self):
        for i in range(len(self.graph)):
            print(i, self.graph[i])


    ###########################
    ### SybilSCAR UTILS #######
    ###########################

    def run_lbp(self, post_list, withBuffer=False):
        """ 
        run LinLBP one iteration with 'graph_list' & 'prior_list'
        :return:
        """

        
        graph_list = self.graph
        node_num = self.node_num
        prior_list = self.prior_list

        wei = Constants.weight
        if withBuffer:
            wei *= Constants.buffer

        next_post_list = [0.0] * node_num

        # graph
        # 1: [4, 2, 3]
        # 2: [3, 4, 5]
        for nei_list in enumerate(graph_list):
            #score_tmp = float(post_list[nei_list[0]])
            score_tmp = 0
            prior_score = prior_list[nei_list[0]]

            for nei in nei_list[1]:
                post_score = post_list[int(nei)]
                score_tmp += (2.0 * wei * post_score)

            score_tmp += prior_score

            if score_tmp > 0.5:
                score_tmp = 0.5
            if score_tmp < -0.5:
                score_tmp = -0.5

            next_post_list[nei_list[0]] = score_tmp

        post_list = copy.deepcopy(next_post_list)

        return post_list

