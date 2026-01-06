# Decoding signal geometry example

Small example of analyzing decoded signal geometry with single hidden layer mlp trained on noisy XOR task with irrelevant random input bit.  The representations analyzed are the hidden layer activations in response to 2000 probe inputs.  

## Requirements

- Python >= 3.12.7

```
pip install -r requirements.txt
```

**To train the multi-layer perceptron and extract more representations:**
```
python train_mlp_xor.py n_hidden load_data seed
```
where n_hidden is the number of units in the hidden layer, load_data = True will load a previously-generated noisy xor dataset with a third irrelevant random bit.  This dataset has 10000 training points and 2000 test points.  Seed specifies the pytorch seed.  

This will save the hidden layer activations to a .pt file ```trained_activations/mlp_xor_activations_seed{seed}_nhidden{n_hidden}_ntest2000.pt```

**To view and analyze previously extracted representations:**

Use the jupyter notebook make_xor_figs.ipynb.  This notebook will load some previously extracted representations included in the ```trained_activations``` folder.  
