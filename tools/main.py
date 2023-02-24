import argparse
import os
import json
from pathlib import Path

import uinauil


def dir_path(path):
    if os.path.isdir(path):
        return path
    elif os.path.isdir(Path(path).parent):
        os.mkdir(path)
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Uinauil, the benchmark for Italian NLP tasks.')
    parser.add_argument('--task', type=str, help='NLP task ID', required=True,
                        choices=['haspeede', 'textualentailment', 'eventi', 'sentipolc','facta','ironita'])
    parser.add_argument('--data-folder', type=dir_path, help='Download benchmarks to this directory',
                        default="./uinauil_data")
    parser.add_argument('--predictions', type=argparse.FileType('r'),
                        help='Evaluate these predictions against task gold standard', required=False)

    args = parser.parse_args()

    print(args)

    task = args.task
    target = args.data_folder

    dr = uinauil.DataReader(task, target);

    if args.predictions is not None:
        benchmark = uinauil.Task(task)
        with args.predictions as file:
            data = json.load(file)
            print(data)
            scores = benchmark.evaluate(data)
            print(scores)
