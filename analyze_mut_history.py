from pathlib import Path

curr_file = Path(__file__).resolve()
main_dir = curr_file.parent
result_dir = main_dir / "result"

target_dir = result_dir / "241130-v1"
target_file = target_dir / "mutation_history_Facebook.csv"

mutation_history = []
mutation_history_dict = {}

with open(target_file, "r") as f:
    lines = f.readlines()
    for line in lines[1:]:
        info = line.strip().split(",")
        iter = int(info[0])
        mutation_str = info[1]
        mutation_new_str = mutation_str[:-2]

        mutation_history.append(mutation_new_str)
        
        if mutation_new_str not in mutation_history_dict:
            mutation_history_dict[mutation_new_str] = 0
        mutation_history_dict[mutation_new_str] += 1

import json
print(json.dumps(mutation_history_dict, indent=4))

# plot a graph of mutation history
import matplotlib.pyplot as plt

# plot a graph key - value
# x-axis: mutation operation
# y-axis: counts of mutation operation
plt.bar(mutation_history_dict.keys(), mutation_history_dict.values())
plt.xlabel("Mutation Operation")
plt.ylabel("Counts")
plt.savefig(target_dir / "mutation_history_Facebook.png")