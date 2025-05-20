import argparse
import os
import os.path as osp
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import re

def parse_configuration(path: str) -> None:
    # 1) dataset+strategy section (parent dir of the last segment)
    dataset_section = os.path.basename(os.path.dirname(path.rstrip("/")))

    # 2) dataset is what comes after "__", before the first "_"
    dataset = dataset_section.split("__")[-1].split("_")[0]

    # 3) strategy is the first "strat_<number>"
    strat_match = re.search(r"(strat_[0-9]+)", dataset_section)
    strategy = strat_match.group(1) if strat_match else "N/A"

    # 4) now parse the final segment
    last = os.path.basename(path.rstrip("/"))
    pattern = re.compile(
        r"^(?P<version>v[\d-]+)_"         # e.g. v2-0_
        r"(?P<n_trials>\d+)trials_"       # e.g. 1trials_
        r"(?P<n_samples>[\d_]+)keep_"     # e.g. 5_1keep_
        r"(?P<n_keep>[\d_]+)samples_"     # e.g. 25_250samples_
        r"l(?P<img_size>\d+)$"            # e.g. l1
    )
    m = pattern.match(last)
    if not m:
        raise ValueError(f"Could not parse configuration from '{last}'")

    # normalize version (replace '-' with '_', if you prefer)
    version   = m.group("version").replace("-", "_")
    n_samples = m.group("n_samples")
    n_keep    = m.group("n_keep")
    img_size  = m.group("img_size")

    # 5) print out
    print("CONFIGURATION:")
    print(f"dataset:     {dataset}")
    print(f"strategy:    {strategy}")
    print(f"SD version:  {version}")
    print(f"n_samples:   {n_samples}")
    print(f"n_keep:      {n_keep}")
    print(f"img_size:    {img_size}")
    return 
    
def top_n_accuracy(topn, k):
    count = 0
    top_k_count = 0

    for correct, preds in topn.items():
        if len(preds) > 1:
            for pred in preds:  
                topk = pred[:k]
                if correct in topk:
                    top_k_count += 1
                count += 1
        else:
            topk = preds[:k]
            if correct in topk:
                top_k_count += 1
            count += 1
    # print(f"topk count: {count}")
    return top_k_count/count

def mean_std_per_class_acc(correct, labels):
    acc_per_class = []
    
    unique_classes = torch.unique(labels)
    for cls in unique_classes:
        mask = labels == cls
        acc = correct[mask].sum().float() / mask.sum().float()
        acc_per_class.append(acc.item())

    acc_per_class = torch.tensor(acc_per_class)
    mean_acc = acc_per_class.mean().item()
    std_acc = acc_per_class.std(unbiased=True).item()

    return mean_acc, std_acc

def mean_per_class_acc(correct, labels):
    """
    Calculate the mean per-class accuracy given the correct predictions and labels.

    Args:
        correct (Tensor): Tensor of correct predictions.
        labels (Tensor): Tensor of class labels.

    Returns:
        float: Mean per-class accuracy.

    Raises:
        ZeroDivisionError: If there are no instances of a class in the labels.
    """
    total_acc = 0
    for cls in torch.unique(labels):
        mask = labels == cls
        total_acc += correct[mask].sum() / mask.sum()
    return total_acc / len(torch.unique(labels))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str)

    args = parser.parse_args()
    # parse_configuration(args.folder)

    # get list of files
    files = os.listdir(args.folder)
    files = sorted([f for f in files if f.endswith('.pt')])

    preds = []
    labels = []
    inf = []
    topn = {}
    for f in tqdm(files):
        data = torch.load(osp.join(args.folder, f))

        preds.append(data['pred'])
        labels.append(data['label'])
        inf.append(data['inf_time'])
        if len(data) > 4:
            print(f"{data['label']}: {data['topn'][:5]} : {data['inf_time']}")
            # print(f"{data['label']}: {data['topn'][:5]}")
            if data['label'] in topn:
                topn[data['label']] += [data['topn']]
            else:
                topn[data['label']] = [data['topn']]            
        # else:
            # print(f"{data['label']} : {data['pred']}")

    print(f"count (test images): {len(preds)}")

    preds = torch.tensor(preds)
    labels = torch.tensor(labels)
    inf = torch.tensor(inf)

    correct = (preds == labels).sum().item()
    print(f'Mean per class acc: {mean_per_class_acc(preds == labels, labels) * 100:.2f}%')
    mean, std = mean_std_per_class_acc(preds == labels, labels)
    # std_low = mean - std
    # std_high = mean + std
    print(f'Std: { std} ')

    print(f'Mean inference time: {inf.sum() / len(preds) :.2f} sec')

    print(f'Top 1 acc: {correct / len(preds) * 100:.2f}%')

    if len(data) > 4:
        print(f'Top 3 acc: {top_n_accuracy(topn, 3) * 100:.2f}%')
        print(f'Top 2 acc: {top_n_accuracy(topn, 2) * 100:.2f}%')
        print(f'Top 5 acc: {top_n_accuracy(topn, 5) * 100:.2f}%')

if __name__ == '__main__':
    main()
