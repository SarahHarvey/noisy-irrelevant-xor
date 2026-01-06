import torch
from torch.utils.data import DataLoader, TensorDataset

import torch.nn as nn
import torch.optim as optim
import argparse
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train MLP on XOR task')
parser.add_argument('--n_hidden', type=int, default=6, help='Number of hidden units')
parser.add_argument('--seed', type=int, default=None, help='Random seed (optional)')
parser.add_argument('--load_data', action='store_true', help='Load existing data instead of generating new set')
args = parser.parse_args()

n_hidden = args.n_hidden

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")

if args.seed is not None:
    seed = args.seed
    torch.manual_seed(seed)
else:
    seed = None

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create noisy XOR data
n_samples = 10000
n_test = 2000
noise_std = 0.1

if args.load_data:
    data = torch.load("data/mlp_xor_dataset.pt")
    X_train = data["train_set"]["X_train"].to(device)
    y_train = data["train_set"]["y_train"].to(device)
    bits = data["bits_train"].to(device)
    X_clean = data["clean_set"]["X_clean"].to(device)
    y_clean = data["clean_set"]["y_clean"].to(device)
    X_test = data["test_set"]["X_test"].to(device)
    y_test = data["test_set"]["y_test"].to(device)
    bits_test = data["bits_test"].to(device)
    n_samples = X_train.shape[0]
    n_test = X_test.shape[0]
    
else:

    # Create noisy XOR data
    n_samples = 10000
    n_test = 2000
    noise_std = 0.1
    # Generate random binary inputs
    bits = torch.randint(0, 2, (n_samples, 3)).float()
    # Add Gaussian noise around the correct values
    X_train = (bits + torch.randn(n_samples, 3) * noise_std).to(device)
    # XOR labels (based on clean bits)
    y_train = ((bits[:, 0] != bits[:, 1]).float().unsqueeze(1)).to(device)

    # Keep clean XOR points for visualization
    X_clean = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32).to(device)
    y_clean = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32).to(device)

    # Generate test set 
    bits_test = torch.randint(0, 2, (n_test, 3)).float()
    X_test = (bits_test + torch.randn(n_test, 3) * noise_std).to(device)
    y_test = ((bits_test[:, 0] != bits_test[:, 1]).float().unsqueeze(1)).to(device)

# Create data loader
dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, loss function, and optimizer
model = MLP(input_size=3, hidden_size=n_hidden, output_size=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Hook to save activations after ReLU
activations = {}
activations["seed"] = seed
activations["noise_std"] = noise_std
activations["n_samples"] = n_samples
activations["n_test"] = n_test
activations["n_hidden"] = n_hidden
activations["test_set"] = {"X_test": X_test, "y_test": y_test}

def get_activation(name):
    def hook(module, input, output):
        activations[name] = output.detach().cpu()
    return hook

# Register hook on the ReLU layer
hook_handle = model.relu.register_forward_hook(get_activation('relu'))
hook_handle_2 = model.fc2.register_forward_hook(get_activation('fc2'))
hook_handle_1 = model.fc1.register_forward_hook(get_activation('fc1'))

# Training loop
epochs = 1000
for epoch in range(epochs):
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.8f}")

# Test the model on clean XOR points
with torch.no_grad():
    predictions = torch.sigmoid(model(X_test))
    predicted_labels = (predictions > 0.5).float() #predictions.float()
    print("\nXOR Results (test inputs):")
    for i in range(4):
        print(f"Input: {X_test[i].cpu().numpy()} -> Predicted: {predicted_labels[i].item():.2f}, Actual: {y_test[i].item():.0f}")
    # Print prediction accuracy
    correct = (predicted_labels.round() == y_test).sum().item()
    accuracy = correct / n_test * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

# Get activations for clean inputs for visualization
with torch.no_grad():
    _ = model(X_test)

# Print activations after ReLU for some test inputs
print("\nReLU Activations for each test input:")
for i in range(4):
    print(f"Input: {X_test[i].cpu().numpy()} -> ReLU activations: {activations['relu'][i].numpy()}")

# Save model weights
activations["model_weights"] = {name: param.detach().cpu() for name, param in model.named_parameters()}

# Clean up hook
hook_handle.remove()
hook_handle_1.remove()
hook_handle_2.remove()

# Check if directory exists, if not create it

if not os.path.exists("trained_activations"):
    os.makedirs("trained_activations")

torch.save(activations, f"trained_activations/mlp_xor_activations_seed{str(seed)}_nhidden{n_hidden}_ntest{n_test}.pt")