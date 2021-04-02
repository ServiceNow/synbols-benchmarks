![#Synbols](https://github.com/ElementAI/synbols/raw/master/title.png)
# Probing Learning Algorithms with Synthetic Datasets

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

![Synbols](https://github.com/ElementAI/synbols-benchmarks/raw/master/cover.png)

## Description

This repository contains the code for reproducing experiments in [1].

To use the Synbols tool for generating new datasets, please visit https://github.com/ElementAI/synbols

## Pytorch Dataset Quick Setup
```python
from pytorch_examples.datasets import Synbols
from torchvision import transforms as tt

dataset_path = "./"
dataset_name = "default_n=100000_2020-Oct-19.h5py"

synbols = Synbols(args.data_path,
                  dataset_name=args.dataset)
train_dataset = synbols.get_split('train', tt.ToTensor())
val_dataset = synbols.get_split('val', tt.ToTensor())
```

For a complete example run `./pytorch_examples/minimal_classification.py` from the root folder of this project:

```bash
python -m pytorch_examples.minimal_classification
```

It should reach >70% accuracy.

## Bibliography


[1] Lacoste, A., Rodríguez, P., Branchaud-Charron, F., Atighehchian, P., Caccia, M., Laradji, I., Drouin, A., Craddock, M., Charlin, L. and Vázquez, D., 2020. [Synbols: Probing Learning Algorithms with Synthetic Datasets.](https://arxiv.org/abs/2009.06415)
