
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


#1. PRUNABLE LINEAR LAYER
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.ones(out_features, in_features) * -0.5)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        if not self.training:
            gates = (gates > 0.5).float()
        pruned_weights = self.weight * gates
        return nn.functional.linear(x, pruned_weights, self.bias)



# 2. SELF PRUNING NETWORK
class SelfPruningNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            PrunableLinear(3072, 512),
            nn.ReLU(),
            PrunableLinear(512, 256),
            nn.ReLU(),
            PrunableLinear(256, 128),
            nn.ReLU(),
            PrunableLinear(128, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)

    def get_all_gates(self):
        gates = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates.append(torch.sigmoid(module.gate_scores))
        return gates



# 3. LOSS FUNCTION
def compute_total_loss(outputs, labels, model, lambda_sparse):
    classification_loss = nn.CrossEntropyLoss()(outputs, labels)
    all_gates = model.get_all_gates()
    sparsity_loss = sum(gates.mean() for gates in all_gates) / len(all_gates)
    total_loss = classification_loss + lambda_sparse * sparsity_loss
    return total_loss, classification_loss, sparsity_loss



# 4. DATA LOADING
def get_dataloaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader



# 5. TRAINING LOOP
def train_model(lambda_sparse, train_loader, device, epochs=20):
    model = SelfPruningNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"\n{'='*50}")
    print(f"Training with lambda = {lambda_sparse}")
    print(f"{'='*50}")

    for epoch in range(epochs):
        model.train()
        running_total = 0.0
        running_cls = 0.0
        running_sparse = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            total_loss, cls_loss, sparse_loss = compute_total_loss(
                outputs, labels, model, lambda_sparse
            )
            total_loss.backward()
            optimizer.step()

            running_total += total_loss.item()
            running_cls += cls_loss.item()
            running_sparse += sparse_loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        print(f"Epoch [{epoch+1:2d}/{epochs}] "
              f"Total: {running_total/len(train_loader):.4f} | "
              f"Cls: {running_cls/len(train_loader):.4f} | "
              f"Sparse: {running_sparse/len(train_loader):.4f} | "
              f"Train Acc: {100.*correct/total:.2f}%")

    return model


# 6. EVALUATION
def evaluate_model(model, test_loader, lambda_val, device, threshold=0.5):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    test_accuracy = 100. * correct / total
    all_gates = model.get_all_gates()
    total_gates = sum(gates.numel() for gates in all_gates)
    pruned_gates = sum((gates < threshold).sum().item() for gates in all_gates)
    sparsity = 100. * pruned_gates / total_gates

    print(f"\nLambda: {lambda_val}")
    print(f"Test Accuracy:  {test_accuracy:.2f}%")
    print(f"Sparsity Level: {sparsity:.2f}%")
    print(f"Total Gates:    {total_gates:,}")
    print(f"Pruned Gates:   {pruned_gates:,}")

    return test_accuracy, sparsity



# 7. GATE DISTRIBUTION PLOT
def plot_gate_distribution(trained_models, lambdas, results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, lam in enumerate(lambdas):
        model = trained_models[lam]
        all_gate_values = []
        for gates in model.get_all_gates():
            all_gate_values.append(gates.detach().cpu().numpy().flatten())
        all_gate_values = np.concatenate(all_gate_values)

        ax = axes[idx]
        ax.hist(all_gate_values, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
        ax.axvline(x=0.5, color="red", linestyle="--", label="Threshold (0.5)")
        ax.set_title(
            f"Lambda = {lam}\n"
            f"Sparsity: {results[lam]['sparsity']:.1f}% | "
            f"Acc: {results[lam]['accuracy']:.1f}%"
        )
        ax.set_xlabel("Gate Value")
        ax.set_ylabel("Frequency")
        ax.legend()

    plt.suptitle("Gate Value Distribution After Training", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("gate_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Plot saved as gate_distribution.png")



# 8. MAIN
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders()

    lambdas = [0.1, 0.3, 0.7]
    trained_models = {}
    results = {}

    for lam in lambdas:
        trained_models[lam] = train_model(lam, train_loader, device, epochs=20)

    for lam in lambdas:
        acc, sparsity = evaluate_model(trained_models[lam], test_loader, lam, device)
        results[lam] = {"accuracy": acc, "sparsity": sparsity}

    plot_gate_distribution(trained_models, lambdas, results)
