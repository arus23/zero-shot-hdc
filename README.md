<div align="center">

<!-- TITLE -->
# **Just Leaf It: Accelerating Diffusion Classifiers with Hierarchical Class Pruning**

[![arXiv](https://img.shields.io/badge/cs.LG-arXiv:2303.16203-b31b1b.svg)](https://arxiv.org/abs/2411.12073)
</div>

This repository is the official codebase for [Just Leaf It: Accelerating Diffusion Classifiers with Hierarchical Class Pruning](https://arxiv.org/abs/2411.12073) by Arundhati S. Shanbhag, Brian B. Moser, Tobias C. Nauen, Stanislav Frolov, Federico Raue, Andreas Dengel.
<!-- DESCRIPTION -->

## Installation
Create a virtual environment using the command:
```bash
python3 -m venv 
```

Activate the virtual environment:
```bash
source venv/bin/activate 
```
Install the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Hierarchical Diffusion Classifier

```bash
python eval_hdc.py --dataset cifar100 \
  --to_keep 5 1 --n_samples 50 500 --loss l1 \
  --prompt_path prompts/cifar100_prompts.csv \
  --strategy 1 \
  --info_dir hierarchies/cifar100/ --root_wnid "n0000"\
```
This command reads prompts from a csv file in the `prompts/` folder and perform hierarchical zero-shot classification using the hierarchy structure specified in `hierarchies/cifar100/`. The strategy can be set to 1 or 2 for fixed and adaptive respectively. Specifiying the root node is essential for hierarchical diffusion classifier to identify the root of the tree.

Similar to the original codebase, the losses are saved separately for each test image in the log directory. For the command above, the log directory is `data/cifar10/v2-0_1trials_5_1keep_50_500samples_l1`. Accuracy can be computed by running:
```bash
python scripts/print_acc.py data/cifar10/v2-0_1trials_5_1keep_50_500samples_l1_s1
```

Commands to run Diffusion Classifier on each dataset are [here](commands.md). 
If evaluation on your use case is taking too long, there are a few options: 
1. Parallelize evaluation across multiple workers. Try using the `--n_workers` and `--worker_idx` flags.
2. Play around with the evaluation strategy (e.g. `--n_samples` and `--to_keep`).
3. Evaluate on a smaller subset of the dataset. Saving a npy array of test set indices and using the `--subset_path` flag can be useful for this.

## Citation

If you find this work useful in your research, please cite:

```bibtex
@misc{shanbhag2025justleafitaccelerating,
      title={Just Leaf It: Accelerating Diffusion Classifiers with Hierarchical Class Pruning}, 
      author={Arundhati S. Shanbhag and Brian B. Moser and Tobias C. Nauen and Stanislav Frolov and Federico Raue and Andreas Dengel},
      year={2025},
      eprint={2411.12073},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.12073}, 
}

```