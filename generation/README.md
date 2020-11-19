## Reproducing generation results
### Setup
Reproduced with a fresh installation of `python3.8`

1. Install requirements

```pip install -r requirements.txt```

### Reproduce classification experiments
1. Launch all the experiments (will sequentially run all of them)

```python3 trainval.py --savedir_base </experiments/directory/path> --exp_group_list <'deepinfomax' or 'vae'>```

Results should be in `<experiments/directory/path>` (without the brackets)
  
### Run a specific experiment
To run a specific experiment edit the [exp_config](https://github.com/ElementAI/synbols-benchmarks/blob/master/classification/exp_configs.py)

### Visualize results
1. Start a [jupyter notebook](https://jupyter.org/)
2. Open the [notebook](https://github.com/ElementAI/synbols-benchmarks/blob/master/classification/Visualize%20Results.ipynb)
3. Add the path where experiments were saved (`<experiments directory>`)
4. Run cell and visualize

### Questions
Contact Pau Rodriguez for questions or open an issue and tag @prlz77
