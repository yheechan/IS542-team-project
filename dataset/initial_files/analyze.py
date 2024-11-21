from pathlib import Path
import sys

args = sys.argv
if len(args) != 2:
    print("Usage: python3 analyze.py <Facebook|Enron>")
    sys.exit(1)
    
target_subject = args[1]

if target_subject not in ["Facebook", "Enron"]:
    print("Invalid target subject")
    sys.exit(1)

code_file = Path(__file__).resolve()
cwd = code_file.parent
target_file = cwd / target_subject / "prior_0(0.5).txt"

nodes = []
scores = []
zero_nodes = []
benign_nodes = []
sybil_nodes = []
with open(target_file, "r") as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        
        data = line.split(" ")
        node = int(data[0])
        score = float(data[1])

        scores.append(score)
        if node not in nodes:
            nodes.append(node)
        
        if score == 0.0:
            zero_nodes.append(node)
        elif score < 0.0:
            benign_nodes.append(node)
        else: # score > 0.0
            sybil_nodes.append(node)


print(f"Total number of scores: {len(scores)}")
print(f"Total nodes: {len(nodes)}")
print(f"Total zero nodes: {len(zero_nodes)}")
print(f"Total benign nodes: {len(benign_nodes)}")
print(f"Total sybil nodes: {len(sybil_nodes)}")
