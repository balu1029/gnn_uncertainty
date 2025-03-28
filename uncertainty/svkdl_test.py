import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import gpytorch
import math
from sklearn.decomposition import PCA


# Generate a more complex toy regression dataset
np.random.seed(42)
batch_size = 32
num_batches = 20
num_epochs = 500
noise = 0
hidden_size = 2
num_inducing_points = 8
X = 4 * np.random.rand(num_batches, batch_size, 1)
y = (
    4
    + 3 * X
    + np.sin(1.5 * np.pi * X)
    + noise * np.random.randn(num_batches, batch_size, 1)
)

X_train = 1.25 * np.random.rand(int(num_batches), batch_size, 1) + 0.25
X_test = np.linspace(0, 2, batch_size * num_batches).reshape(num_batches, batch_size, 1)

y_train = (
    4
    + 3 * X_train
    + np.sin(1.5 * np.pi * X_train)
    + noise * np.random.randn(*X_train.shape)
)
y_test = (
    4
    + 3 * X_test
    + np.sin(1.5 * np.pi * X_test)
    + noise * np.random.randn(*X_test.shape)
)

X_ground_truth = np.linspace(0, 2, 100)
y_ground_truth = 4 + 3 * X_ground_truth + np.sin(1.5 * np.pi * X_ground_truth)

# Split the dataset into training and testing sets
"""X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)"""

# Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


# Define a larger neural network model
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.hidden1 = nn.Linear(1, 40)
        self.hidden2 = nn.Linear(40, 20)
        self.output = nn.Linear(20, hidden_size)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x


class GPRegressionModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super(GPRegressionModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SVGPRegressionModel(nn.Module):
    def __init__(self, inducing_points):
        super(SVGPRegressionModel, self).__init__()
        self.feature_extractor = RegressionModel()
        self.gp_layer = GPRegressionModel(inducing_points)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.gp_layer(features.squeeze(-1))


# Initialize inducing points for the GP layer
inducing_points = torch.tensor(
    np.random.randn(num_inducing_points, hidden_size), dtype=torch.float32
)

# Create the SVGP model
svgp_model = SVGPRegressionModel(inducing_points)
likelihood = gpytorch.likelihoods.GaussianLikelihood()

# Define the optimizer for the SVGP model
svgp_optimizer = torch.optim.Adam(
    [
        {"params": svgp_model.feature_extractor.parameters(), "lr": 0.01},
        {"params": svgp_model.gp_layer.parameters(), "lr": 0.01},
        {"params": likelihood.parameters(), "lr": 0.01},
    ],
    lr=0.01,
)

# Define the loss function for the SVGP model
mll = gpytorch.mlls.VariationalELBO(
    likelihood, svgp_model.gp_layer, num_data=X_train_tensor.flatten().size(0)
)

# Train the SVGP model
svgp_model.train()
likelihood.train()

for epoch in range(num_epochs):
    for x, target in zip(X_train_tensor, y_train_tensor):
        x = x.unsqueeze(0)
        target = target.unsqueeze(0)

        svgp_optimizer.zero_grad()
        output = svgp_model(x)
        loss = -mll(output, target.squeeze(-1))
        loss.backward()
        svgp_optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Predict on the test set using the SVGP model
svgp_model.eval()
likelihood.eval()
with torch.no_grad():
    y_pred_svgp = likelihood(svgp_model(X_test_tensor))


# Convert predictions to numpy array
y_mean = y_pred_svgp.mean.numpy().flatten()
y_uncertainty = y_pred_svgp.variance.sqrt().numpy().flatten()
# Sort the test set and predictions for better visualization
X_test_tensor = X_test_tensor.flatten()
y_test_tensor = y_test_tensor.flatten()
sorted_indices = X_test_tensor.squeeze().argsort()
X_test = X_test_tensor.squeeze()[sorted_indices]
y_test = y_test_tensor.squeeze()[sorted_indices]
y_mean = y_mean[sorted_indices]
y_uncertainty = y_uncertainty[sorted_indices]

# Visualize the results
plt.scatter(
    X_train_tensor.flatten(),
    y_train_tensor.flatten(),
    color="orange",
    label="Training Points",
)
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.scatter(X_test, y_mean, color="red", label="Predicted")
plt.plot(X_ground_truth, y_ground_truth, color="green", label="Ground Truth")
plt.fill_between(
    X_test.flatten(),
    y_mean - 2 * y_uncertainty,
    y_mean + 2 * y_uncertainty,
    color="red",
    alpha=0.5,
    label="Uncertainty",
)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Non-linear Regression on Toy Dataset")
plt.legend()
plt.show()
plt.close()

# Plot the embedding of X_test with a colormap following the numbers in X_test

# Extract features from the test set using the feature extractor
with torch.no_grad():
    test_features = svgp_model.feature_extractor(X_test_tensor.unsqueeze(-1))
    train_features = svgp_model.feature_extractor(X_train_tensor.unsqueeze(-1))


# Perform PCA to reduce the dimensionality to 2D for visualization
def pca(features, n_components=2):
    pca_model = PCA(n_components=n_components)
    features_2d = pca_model.fit_transform(features.detach().numpy())
    return torch.tensor(features_2d, dtype=torch.float32)


test_features = pca(test_features.view(-1, hidden_size))
train_features = pca(train_features.view(-1, hidden_size))
test_features_2d = test_features.numpy()
train_features_2d = train_features.numpy()


# Get inducing points from the variational strategy
inducing_points = (
    svgp_model.gp_layer.variational_strategy.inducing_points.detach().numpy()
)
# Create a scatter plot of the 2D features with a colormap
plt.scatter(
    inducing_points[:, 0],
    inducing_points[:, 1],
    color="red",
    marker="x",
    label="Inducing Points",
)
plt.scatter(train_features_2d[:, 0], train_features_2d[:, 1], color="orange")
plt.scatter(
    test_features_2d[:, 0],
    test_features_2d[:, 1],
    c=X_test.numpy(),
    cmap="viridis",
    alpha=0.2,
)

plt.colorbar(label="X_test values")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("2D Embedding of X_test with Colormap")
plt.show()
