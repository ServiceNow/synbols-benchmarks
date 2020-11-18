## Reproducing classification results
### Setup
Reproduced with a fresh installation of `python3.8`

1. Install requirements

```pip install -r requirements.txt```

### Reproduce classification experiments
1. Launch all the experiments (will sequentially run all of them)
`python3 trainval.py --savedir_base <experiments directory> --exp_group_list baselines`

Results should be in `<experiments directory>`
  
### Run a specific experiment
To run a specific experiment edit the [exp_config](https://github.com/ElementAI/synbols-benchmarks/blob/master/classification/exp_configs.py)


### Questions
Contact Pau Rodriguez for questions or open an issue and tag @prlz77
