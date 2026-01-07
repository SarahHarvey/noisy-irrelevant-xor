# Decoding signal geometry example


Small example of analyzing decoded signal geometry with single hidden layer MLP trained on noisy XOR task with irrelevant random input bit.

  
## Requirements

  
- Python >= 3.12.7

  
```bash
pip install -r requirements.txt
```


## To view and analyze previously extracted representations:

  
Use the jupyter notebook "rep_sim_analysis.ipynb". This notebook will load some previously extracted representations included in the ```trained_activations``` folder, and perform some basic plotting and analysis.

  
## To train a new MLP XOR model


### Overview


`train_mlp_xor.py` trains a simple multi-layer perceptron (MLP) on a noisy XOR classification task. The model learns to predict XOR from two input bits while ignoring a third randomly generated irrelevant bit. All three inputs are corrupted with Gaussian noise.

  
### Usage


```bash

python train_mlp_xor.py [OPTIONS]

```


### Arguments


| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--n_hidden` | int | 6 | Number of hidden units in the MLP |
| `--seed` | int | None | Random seed for reproducibility |
| `--load_data` | flag | False | Load pre-existing dataset from `data/mlp_xor_dataset.pt` |

  
### Examples


```bash

# Train with default settings (6 hidden units, random seed)

python train_mlp_xor.py

# Train with 128 hidden units and a fixed seed

python train_mlp_xor.py --n_hidden 128 --seed 42

# Train using pre-saved dataset

python train_mlp_xor.py --n_hidden 32 --seed 33 --load_data

```


### Output

  
The script saves activations and model weights to:

```
trained_activations/mlp_xor_activations_seed{SEED}_nhidden{N_HIDDEN}_ntest2000.pt
```

The saved file contains:

- `seed`, `noise_std`, `n_samples`, `n_test`, `n_hidden`: Training configuration
- `test_set`: Test inputs and labels
- `relu`, `fc1`, `fc2`: Layer activations on test set
- `model_weights`: Trained model parameters


### Model Architecture

  

- **Input**: 3 features (2 XOR bits + 1 irrelevant bit, with Gaussian noise σ=0.1)
- **Hidden**: Fully connected layer with ReLU activation
- **Output**: Single logit for binary classification (BCEWithLogitsLoss)


### Training Details

  
- **Training samples**: 10,000

- **Test samples**: 2,000

- **Batch size**: 32

- **Optimizer**: Adam (lr=0.01)

- **Epochs**: 1,000


## To run on the Flatiron Institute HPC cluster:


Use the SLURM job configuration .sh file ```train_mlp_xor.sh```. Example:

  
``` sbatch train_mlp_xor.sh 5 True 33 ```