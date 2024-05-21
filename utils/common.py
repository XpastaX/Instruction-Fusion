import torch
import random
import os
import numpy as np
import json


def check_dir(path, creat=True, force=False):
    path = os.path.split(path)[0]
    if not os.path.exists(path):
        if creat:
            os.makedirs(path)
            print('Folder %s has been created.' % path)
            return True
        else:
            return False
    else:
        if force:
            os.makedirs(path)
            print('Force to create %s.' % path)
        return True


def print_args(args):
    for _arg in args._get_kwargs():
        print(f"{_arg[0]}:{_arg[1]}")


def set_seed(seed):
    """
    :param seed:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def write_log(txt, path, prt=True, creat=False):
    if prt:
        print(txt)
    if txt[-1:] != '\n':
        txt += '\n'
    if not creat:
        with open(path, 'a') as file:
            file.writelines(txt)
    else:
        with open(path, 'w') as file:
            file.writelines(txt)


def print_class(obj):
    tmp = {name: value for name, value in obj.__dict__.items() if '__' not in name}
    txt = ''
    for key in tmp:
        txt += f'{key}:{tmp[key]}\n'
    return txt


def load_data(path):
    try:
        return json.load(open(path, 'r'))
    except:
        with open(path, 'r') as f:
            data = []
            for line in f:
                data.append(json.loads(line))
    return data
