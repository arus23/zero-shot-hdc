import argparse
import os
import os.path as osp
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd

def top_n_accuracy(topn, k):
    count = 0
    top_k_count = 0

    for correct, preds in topn.items():
        topk = preds[:k]
        if len(topk) < 3:
            continue
        if correct in topk:
            top_k_count += 1
        count += 1
    print(f"topk count: {count}")
    return top_k_count/count

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
            topn[data['label']] = data['topn'][:5]

    print(f"count (test images): {len(preds)}")

    preds = torch.tensor(preds)
    labels = torch.tensor(labels)
    inf = torch.tensor(inf)

    correct = (preds == labels).sum().item()
    print(f'Top 1 acc: {correct / len(preds) * 100:.2f}%')

    if len(data) > 4:
        print(f'Top 3 acc: {top_n_accuracy(topn, 3) * 100:.2f}%')

        print(f'Top 5 acc: {top_n_accuracy(topn, 5) * 100:.2f}%')

    print(f'Mean per class acc: {mean_per_class_acc(preds == labels, labels) * 100:.2f}%')

    print(f'Mean inference time: {inf.sum() / len(preds) :.2f} sec')


if __name__ == '__main__':
    main()
