
import lib.constants as Constants

class Config:
    def __init__(
            self, graph_type
    ):
        self.graph_type = graph_type

        # initializes..
        # self.node_cnt, self.num_negative, self.num_positive,
        self.init_graph_configs()
    
    def init_graph_configs(self,):
        if self.graph_type == "Facebook":
            self.node_cnt = 8078
            self.num_negative = 4039
            self.num_positive = 4039
        else:
            raise Exception("No graph type other than Facebook")

