import argparse

import torch
import torch.nn.functional as F
import torchvision.transforms as tt
from tqdm import tqdm

from .datasets import Synbols


# Define a model
class SmallConvNet(torch.nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 64, 3, 1, 0, bias=False)
        self.bn0 = torch.nn.BatchNorm2d(64)
        self.conv1 = torch.nn.Conv2d(64, 64, 3, 1, 0, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.conv4 = torch.nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn4 = torch.nn.BatchNorm2d(64)
        self.linear = torch.nn.Linear(64, n_classes)

    def forward(self, x):
        for i in range(5):
            conv = getattr(self, f"conv{i}")
            bn = getattr(self, f"bn{i}")
            x = conv(x)
            x = bn(x)
            x = F.leaky_relu(x)
        x = x.mean((2, 3))
        return self.linear(x)


def main(args):
    # Load datasets
    synbols = Synbols(args.data_path,
                      dataset_name=args.dataset)
    model = SmallConvNet(synbols.n_classes)
    model.cuda()
    train_dataset = synbols.get_split('train', tt.ToTensor())
    val_dataset = synbols.get_split('val', tt.ToTensor())
    del(synbols)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    for epoch in range(args.epochs):
        print("Epoch", epoch)
        model.train()
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            x, y = batch
            x = x.cuda()
            y = y.cuda()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

        print("train_loss", float(loss))
        with torch.no_grad():
            model.eval()
            hits = 0
            total = 0
            for batch in tqdm(val_loader):
                x, y = batch
                x = x.cuda()
                y = y.cuda()
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                hits += (logits.argmax(-1) == y).float().sum().item()
                total += x.size(0)

        print("val_loss", float(loss), "val_accuracy", hits / total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Folder where synbols datasets can be found or downloaded")
    parser.add_argument("--dataset", type=str, default="default_n=100000_2020-Oct-19.h5py",
                        help="Version of synbols to load. \
                                  Visit https://github.com/ElementAI/synbols-resources/tree/master/datasets/generated \
                                  for the complete list list")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0004)
    parser.add_argument("--batch_size", type=float, default=64)
    args = parser.parse_args()
    main(args)
