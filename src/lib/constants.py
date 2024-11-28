
"""
'constants.py' is used for global variables
"""

from pathlib import Path

constant_py_file_path = Path(__file__).resolve()

lib_dir_path = constant_py_file_path.parent
src_dir_path = lib_dir_path.parent
main_dir_path = src_dir_path.parent

dataset_dir_path = main_dir_path / "dataset"

result_dir_path = main_dir_path / "result"
if not result_dir_path.exists():
    result_dir_path.mkdir()

# Temporary,, don't know what these constants does
buffer = 0.8
sampling_size = 100
num_unlabel = 0

theta = 0.5
weight = 0.01
iteration = 6
threshold = 0


FN_nodes = []
