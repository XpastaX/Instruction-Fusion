import json
from config import system
import openai
from transformers import AutoTokenizer
import tiktoken
import requests
from tqdm import tqdm
import re


def load_data(path):
    try:
        return json.load(open(path, 'r'))
    except:
        with open(path, 'r') as f:
            data = []
            for line in f:
                data.append(json.loads(line))
    return data


def save_data(data, path, jsonl=False):
    if jsonl:
        with open(path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    else:
        json.dump(data, open(path, 'w'), indent=2)


def read_file(path):
    with open(path, 'r') as f:
        txt = f.read()
    return txt


def get_backward_data(data, addon='BW'):
    data_BW = []
    for sample in data:
        inst = sample["messages"][1]["content"]
        response = sample["messages"][2]["content"]
        id = str(sample['id']) + '||' + addon
        if len(re.findall("```", response)) == 2:
            code_chunk = re.search(r'```([\s\S]*?)```', response).group(1)
            code_chunk = "```" + code_chunk + "```"
            inst_BW = "Write a programming question (no need to provide any additional information, " \
                      "just write the question), and its code implementation is as follows:" + "\n" + \
                      code_chunk
            response_reverse = "Question: " + inst
            messages = [{"role": "system", "content": ""},
                        {"role": "user", "content": inst_BW},
                        {"role": "assistant", "content": response_reverse}]
            sample_BW = {
                "dataset": sample['dataset'],
                "id": id,
                "messages": messages
            }
            data_BW.append(sample_BW)
    return data_BW


def convert_back(data):
    new_data = []
    for index, sample in enumerate(data):
        if sample['output'] is None:
            print(f"Find Empty Output at ID {sample['id']}")
        message = [{"role": "system", "content": ""},
                   {"role": "user", "content": sample['instruction']},
                   {"role": "assistant", "content": sample['output']},
                   ]
        sample['messages'] = message
        new_data.append(sample)
    return new_data


def count_input_ids(query, _tokenizer, _system=system):
    _system = "<<SYS>>\n{{system}}\n<</SYS>>\n\n".replace("{{system}}", _system, 1)
    query = "[INST] {{query}} [/INST] ".replace("{{query}}", query, 1)
    tokens = \
        _tokenizer.encode(system + query, padding=True, truncation=True, return_tensors='pt', add_special_tokens=False)[
            0]
    return len(tokens)


def estimate_bill(prompt=None, response=None, model=None, tqdm_disable=False):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    sum_token_prompt = 0
    sum_token_response = 0
    if prompt is not None:
        print('Tokenizing prompts')
        for text in tqdm(prompt, disable=tqdm_disable):
            sum_token_prompt += len(tokenizer.encode(text))
    if response is not None:
        print('Tokenizing responses')
        for text in tqdm(response, disable=tqdm_disable):
            sum_token_response += len(tokenizer.encode(text))

    print(f"prompt_token:{sum_token_prompt}|response_token:{sum_token_response}")
    print(model)
    price_list = {
        'gpt-4-1106-preview': [0.01, 0.03],
        'gpt-4': [0.03, 0.06],
        'gpt-4-32k': [0.06, 0.12],
        'gpt-3.5-turbo-1106': [0.001, 0.002],
        'gpt-3.5-turbo-instruct': [0.0015, 0.002],

    }
    bill = {}
    for mod in price_list:
        bill[mod] = sum_token_prompt / 1000 * price_list[mod][0] + sum_token_response / 1000 * price_list[mod][1]
    print("{:^23}: {:^10} {:^10}".format('model', 'USD', 'CYN'))
    if model is None:
        for mod in bill:
            print(
                "{:<23}: {:<10} {:<10}".format(mod, str(round(bill[mod], 2)), round(convert_usd_to_rmb(bill[mod]), 2)))
        return bill
    else:
        print("{:<23}: {:<10} {:<10}".format(model, str(round(bill[model], 2)),
                                             round(convert_usd_to_rmb(bill[model]), 2)))
        return bill[model]


def convert_usd_to_rmb(amount):
    # API endpoint for currency conversion
    api_url = "https://api.exchangerate-api.com/v4/latest/USD"

    try:
        # Sending a request to the API
        response = requests.get(api_url)
        data = response.json()

        # Getting the exchange rate for USD to RMB (CNY)
        exchange_rate = data['rates']['CNY']

        # Calculating the converted amount
        converted_amount = amount * exchange_rate

        return converted_amount
    except Exception as e:
        return str(e), None


def get_context_window(model_list):
    num_rounds = len(model_list)
    context_window = [0] * num_rounds
    for i, name in enumerate(model_list):
        if 'gpt-4' in name:
            context_window[i] = 8192
        elif 'gpt-3.5' in name:
            context_window[i] = 4096
        else:
            raise NotImplementedError
    return context_window


def load_tokenizer(path, max_token):
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side='right', truncation_side='left',
                                              model_max_length=max_token)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def check_model(model_name):
    # check model
    model_list = openai.Model.list()['data']
    all_models = [item['id'] for item in model_list]
    if type(model_name) is str:
        if model_name not in all_models:
            print("model not available")
            raise NotImplementedError
        return
    else:
        for name in model_name:
            if name not in all_models:
                print("model not available")
                raise NotImplementedError


def combine_instruction_input(data):
    instructions = []
    for d in data:
        instruction = d['instruction']
        input_text = d['input']
        if input_text != '':
            instruction += '\n' + input_text
        instructions.append(instruction)
    return instructions


def combine_output(data):
    output = []
    for d in data:
        resp = d['output']
        output.append(resp)
    return output


def combine_all(data):
    backgrounds = []
    for d in data:
        background = ""
        instruction = d['instruction']
        input = d['input']
        response = d['output']
        history = d['history']
        background += instruction + '\n' + input + '\n' + response
        for item in history:
            background += item[0] + '\n' + item[1] + '\n'
        backgrounds.append(background)
    return backgrounds


def load_stop_words(path='data/raw/stop_words.txt'):
    words = []
    with open(path, 'r') as f:
        for line in f:
            words.append(line.strip())
    return words


def stat_input_len(inputs, tokenizer, max_token=4096, regions=20, print_stat=False):
    stat = []
    for prompt in inputs:
        input_len = count_input_ids(prompt, tokenizer)
        stat.append(input_len)

    if print_stat:
        region_size = max_token / regions
        region_counts = [0] * regions
        for value in stat:
            if value < max_token:
                # Determine the region index
                region_index = int(value // region_size)
                region_counts[region_index] += 1
            else:
                print(f'{value} exceed max token limit!')
        print(f"Max input size is {max(stat)}")
        print(f"Min input size is {min(stat)}")
        for i, count in enumerate(region_counts):
            print(f"Region {i + 1} ({i * region_size}-{(i + 1) * region_size}): {count} values")
    return stat


def cal_resp_size(prompts, tokenizer_path, max_token, round_ratio=None, remain=100):
    tokenizer = load_tokenizer(tokenizer_path, max_token)
    stat = stat_input_len(prompts, tokenizer, max_token=max_token)
    output_size_list = []
    for input_size in stat:
        output_size = max_token - input_size - remain
        if output_size > 100:
            output_size_list.append(output_size)
        else:
            print(f"find oversized prompt")
            output_size_list.append(100)

    if round_ratio is None:
        return output_size_list
    all = sum(round_ratio)
    response_size_list = []
    for output_size in output_size_list:
        response_size = [int(output_size / all * ratio) for ratio in round_ratio]
        response_size_list.append(response_size)

    return response_size_list
