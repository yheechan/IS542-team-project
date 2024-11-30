# IS542-team-project

## Genetic Algorithm-Based Adversarial Attack on SybilSCAR
This is a tool in which automatically constructs a social network graph that aims to bypass sybil node detection tools such as SybilSCAR based on Genetic Algorithm.

## Tool Usage
```
usage: main.py [-h] -t TARGET [-s] -e EXPERIMENT_NAME

main.py is to run GA on SybilSCAR

options:
  -h, --help            show this help message and exit
  -t TARGET, --target TARGET
                        target (e.g., Facebook, etc.)
  -s, --use-seed        use seed
  -e EXPERIMENT_NAME, --experiment-name EXPERIMENT_NAME
                        experiment name
```
* flag explanation:
    * ``-t <target-name>``: flag to indicate target dataset.
    * ``-s``: flag that uses initial seed when given.
    * ``-e <experiment-name>``: flag to indicate experiment name.

## Execution step
```
$ time python3 main.py -e experiment-v1 -t Facebook -s
```
* Parameters for the genetic algorithm can be changed within ``src/main.py`` source file.


Last update Nov 30, 2024
