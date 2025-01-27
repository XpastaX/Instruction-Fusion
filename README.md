# Fine-tuning
We only provide the code to fuse instructions. You can use the following GitHub repo for fine-tuning Code Llama:
https://github.com/hiyouga/LLaMA-Factory/tree/main

# Dataset
download it from huggingface:
- https://huggingface.co/datasets/Pasta009/Instruction-Fusion-Code-v1

# Evaluation
Use evalplus to evaluate on python benchmarks:
https://github.com/evalplus/evalplus

Use bigcode-evaluation-harness to evalute the performance on MultiPL-E
https://github.com/bigcode-project/bigcode-evaluation-harness

# Seed Template
{
"dataset": "your dataset name",
"id": 0,
"instruction": "",
}

We use evol-codealpaca-v1 as seeds.
# Performance
| Method             | Size | Open-source | HumanEval | HumanEval+ | MBPP | MBPP+ |
|--------------------|------|-------------|-----------|------------|------|-------|
| gpt-4-1106-preview | -    | -           | 85.4      | 81.7       | 83.0 | 70.7  |
| gpt-3.5-turbo-1106 | -    | -           | 72.6      | 65.9       | 81.7 | 69.4  |
| StarCoder          | 7B   | weight&data | 24.4      | 20.7       | 33.1 | 28.8  |
| Mistral            | 7B   | weight      | 28.7      | 23.2       | 50.1 | 40.9  |
| CodeLlama-Python   | 7B   | weight      | 37.8      | 34.1       | 57.6 | 45.4  |
| WizardCoder-CLP    | 7B   | weight      | 48.2      | 40.9       | 56.6 | 47.1  |
| MagicoderS-CLP     | 7B   | weight&data | 70.7      | 66.5       | 68.4 | 56.6  |
| CodeLlama-Python   | 13B  | weight      | 42.7      | 36.6       | 61.2 | 50.9  |
| StarCoder          | 15B  | weight&data | 34.1      | 29.3       | 55.1 | 46.1  |
| CodeT5+            | 16B  | weight&data | 31.7      | 26.2       | 54.6 | 44.4  |
| CodeGen-Mono       | 16B  | weight&data | 32.9      | 27.4       | 52.6 | 43.6  |
| CodeLlama-Python   | 34B  | weight      | 51.8      | 42.7       | 67.2 | 52.9  |
| WizardCoder-CLP    | 34B  | weight      | 73.2      | 64.6       | 73.2 | 59.9  |
| [IF-CLP](https://huggingface.co/Pasta009/IF-CLP-13B)             | 13B  | weight&data | 73.8      | 69.5       | 71.7 | 61.7  |
| [IF-CL](https://huggingface.co/Pasta009/IF-CL-13B)              | 13B  | weight&data | 74.4      | 68.3       | 69.7 | 59.4  |
| [IF-CLP](https://huggingface.co/Pasta009/IF-CLP-34B)             | 34B  | weight&data | 75.6      | 69.5       | 73.7 | 62.7  |
| [IF-CL](https://huggingface.co/Pasta009/IF-CL-34B)              | 34B  | weight&data | 78.7      | 71.3       | 71.4 | 60.7  |
| [IF-CL-MC](https://huggingface.co/Pasta009/IF-CL-MC-7B)           | 7B   | weight&data | 76.2      | 71.3       | 70.4 | 57.9  |
| [IF-CL-MC](https://huggingface.co/Pasta009/IF-CL-MC-13B)           | 13B  | weight&data | 79.3      | 72.6       | 69.2 | 57.4  |
| [IF-CL-MC](https://huggingface.co/Pasta009/IF-CL-MC-34B)           | 34B  | weight&data | 82.3      | 75.6       | 72.4 | 61.4  |

# Bibtex
```
@inproceedings{guo-etal-2024-instruction,
    title = "Instruction Fusion: Advancing Prompt Evolution through Hybridization",
    author = "Guo, Weidong  and
      Yang, Jiuding  and
      Yang, Kaitong  and
      Li, Xiangyang  and
      Rao, Zhuwei  and
      Xu, Yu  and
      Niu, Di",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.214",
    doi = "10.18653/v1/2024.acl-long.214",
    pages = "3883--3893",
}
```

