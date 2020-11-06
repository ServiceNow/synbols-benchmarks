# Active learning experiments

This folder contains all experiments related to Active learning.

Requirements
* Pytorch >= 1.5
* Haven
* BaaL >= 1.1.0

`pip install -r requirements.txt`

We expect the dataset to be in the folder $DATA_ROOT.

The list of experiments are in `scripts/active_learning.py`

You will need the script in `synbols-benchmarks/utils/` so please add it to your $PYTHONPATH.

`export PYTHONPATH=$PYTHONPATH:$SB_PATH/utils/`

where `$SB_PATH` is the path to the `synbols-benchmarks` repo.
