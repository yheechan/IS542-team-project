import argparse

class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__(
            description="main.py is to run GA on SybilSCAR"
        )
        self.add_argument("-t", "--target", help="target (e.g., Facebook, etc.)", required=True)
        self.add_argument("-s", "--use-seed", help="use seed", action="store_true")
        # self.add_argument("-o", "--original-graph-file", help="original graph file path", required=True)
