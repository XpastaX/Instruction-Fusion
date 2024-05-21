from tqdm import tqdm
from utils.data import load_data
import json
import argparse
from multiprocessing import Pool
from utils.common import check_dir, print_args, set_seed
from utils.data import read_file
from utils.crawl_from_GPT import get_response_from_message as get_response
import random
import os

# prompt template of instruction fusion
prefix = read_file('prefix/hybrid_random.txt')
# system prompt used to collect answers from GPT-4
system = "As a programmer, your task is to write code to address each provided prompt. If a specific programming " \
         "language is not specified, please use Python. Do NOT use pseudo code."


def process_sample(args):
    pair, model_name, dataset, tmp_file = args
    p1, p2 = pair[0]['instruction'], pair[1]['instruction']
    # format prompt using two seed instructions
    prompt = prefix.replace("{{ prompt1 }}", p1)
    prompt = prompt.replace("{{ prompt2 }}", p2)
    message = [{'role': 'user', 'content': prompt}]
    inst = get_response(message, model_name, wait=10, timeout=60)
    resp = None
    # in GPT think the fused instruction is valid, then collect answer.
    if 'INVALID PROMPT' not in inst and inst is not None:
        message = [{'role': 'system', 'content': system},
                   {'role': 'user', 'content': inst}]
        resp = get_response(message, model_name, wait=10, timeout=60)
    result = {"dataset": 'hybrid_rand',
              "id": f"{pair[0]['id']}-{pair[1]['id']}",
              "system": "",
              "instruction": inst,
              "input": "",
              "output": resp,
              "history": [],
              "prompt1": p1,
              "prompt2": p2,
              }
    with open(tmp_file, 'a') as file:
        file.write(json.dumps(result) + '\n')
    return result


def retrieve_samples(pairs, arg):
    args_list = [(pair, arg.model_name, arg.dataset, arg.tmp_file) for index, pair in enumerate(pairs)]
    results = []
    with open(arg.tmp_file, 'w') as file:
        pass
    with Pool(arg.num_workers) as pool:
        # Prepare the pool tasks
        tasks = [pool.apply_async(process_sample, args=(_arg,)) for _arg in args_list]

        # Use tqdm to track progress
        for task in tqdm(tasks, total=len(tasks)):
            result = task.get()
            # check None result
            if result['output'] is not None:
                results.append(result)
    return results


def update_pair_dict(data,pair_dict=None):
    if pair_dict is None:
        pair_dict = {}
    for sample in data:
        ids = sample['id'].split('-')
        s1, s2 = int(ids[0]), int(ids[1])
        if sample['id'] not in pair_dict:
            pair_dict[sample['id']] = True
            pair_dict[f"{str(s2)}-{str(s1)}"] = True
    print(f"Pair_dict updated based on data: {len(data)} samples")
    return pair_dict


def sample_pairs(data, target_size, pair_dict=None):
    if pair_dict is None:
        pair_dict = {}
    pairs = []
    count = 0
    while count < target_size:
        _p = random.sample(data, 2)
        key1 = f"{_p[0]['id']}-{_p[1]['id']}"
        key2 = f"{_p[1]['id']}-{_p[0]['id']}"
        if key1 not in pair_dict:
            pair_dict[key1] = True
            pair_dict[key2] = True
            pairs.append(_p)
            count += 1
    return pairs, pair_dict


def run(arg):
    # data version, used for naming
    dataset = arg.dataset
    # load data
    path = f'data/code/{dataset}.json'
    print(f"Load data from {path}")
    data = load_data(path)
    # load pair dict which records the fusion history
    if arg.pair_recorder_path != 'tmp':
        pair_dict = load_data(arg.pair_recorder_path)
    else:
        pair_dict = {}
    if os.path.isfile(arg.save_path):
        try:
            print(f"Find previous records at {arg.save_path}, start from the checkpoint.")
            results = load_data(arg.save_path)
            pair_dict = update_pair_dict(results, pair_dict)
        except:
            raise Exception(f'Can not load existing record from {arg.save_path}, '
                            f'please validate or rename the file to prevent overwriting')
    else:
        results = []

    current_size = len(results)
    # incase some fusions are invalid, looping to reach the target amount.
    while current_size < arg.target_size:
        target_size = arg.target_size - current_size
        print(f"Current Size: {current_size}, Need {target_size} More...")
        pairs, pair_dict = sample_pairs(data, target_size, pair_dict=pair_dict)
        results += retrieve_samples(pairs, arg)
        current_size=len(results)
    json.dump(results, open(arg.save_path, 'w'), indent=2)
    json.dump(pair_dict, open(arg.pair_recorder_path, 'w'), indent=2)


if __name__ == "__main__":
    set_seed(123)
    parser = argparse.ArgumentParser(description='xxxxx')
    parser.add_argument('--dataset', type=str, default='all', help='seeds')
    parser.add_argument('--target_size', type=int, default=100000)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--model_name', type=str, default='gpt-4-1106-preview')
    parser.add_argument('--save_path', type=str, default='tmp')
    parser.add_argument('--tmp_file', type=str, default='tmp')
    parser.add_argument('--pair_recorder_path', type=str, default='tmp')
    arguments = parser.parse_args()

    name = f"{arguments.dataset}_{arguments.model_name}_{arguments.target_size}"

    if arguments.save_path == 'tmp':
        arguments.save_path = f'data/hybrid/{name}.json'
    if arguments.tmp_file == 'tmp':
        arguments.tmp_file = f'cache/hybrid_{name}.json'
    if arguments.tmp_file == 'tmp':
        arguments.pair_recorder_path = f'cache/hybrid_{name}_pair.json'

    check_dir(arguments.save_path)
    check_dir(arguments.tmp_file)
    print_args(arguments)
    run(arguments)
