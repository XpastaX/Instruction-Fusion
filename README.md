# Fine-tuning
We only provide the code to fuse instructions. You can use the following GitHub repo for fine-tuning Code Llama:
https://github.com/hiyouga/LLaMA-Factory/tree/main

# Evaluation
Use evalplus to evaluate on python benchmarks:
https://github.com/evalplus/evalplus

Use bigcode-evaluation-harness to evalute the performance on MultiPL-E
https://github.com/bigcode-project/bigcode-evaluation-harness

# Released model
1. IF-CLP-MC-34B
- https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1
- Instruction Fusion
- https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K

Coming Soon
IF-CLP-MC 7B, 13B
IF-CLP 13B, 34B
IF-CL 13B, 34B

# Seed Template
{
"dataset": "your dataset name",
"id": 0,
"instruction": "",
}

We use evol-codealpaca-v1 as seeds.
