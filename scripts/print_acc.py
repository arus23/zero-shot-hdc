import argparse
import os
import os.path as osp
import torch
from tqdm import tqdm


def mean_per_class_acc(correct, labels):
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
    for f in tqdm(files):
        data = torch.load(osp.join(args.folder, f))
        preds.append(data['pred'])
        labels.append(data['label'])
        inf.append(data['inf_time'])

    print(f"count: {len(preds)}")
    # print(f"preds: {labels}")
    preds = torch.tensor(preds)
    labels = torch.tensor(labels)
    inf = torch.tensor(inf)
    # top 1
    correct = (preds == labels).sum().item()
    print(f'Top 1 acc: {correct / len(preds) * 100:.2f}%')
    # mean per class accuracy
    print(f'Mean per class acc: {mean_per_class_acc(preds == labels, labels) * 100:.2f}%')
    # mean time/img
    print(f'Mean inference time: {inf.sum() / len(preds) :.2f}%')
    # mean per class time
    # print(f'Mean per class time: {mean_per_class_time(preds == labels, labels) * 100:.2f}%')

if __name__ == '__main__':
    main()
