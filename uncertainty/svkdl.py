from uncertainty.base_uncertainty import BaseUncertainty

import torch
import torch.nn as nn
import numpy as np
import wandb
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gpytorch


class GPLayer(gpytorch.models.ApproximateGP):
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
        super(GPLayer, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SVGPModel(nn.Module):
    def __init__(self, inducing_points, base_model):
        super(SVGPModel, self).__init__()
        self.feature_extractor = base_model
        self.gp_layer = GPLayer(inducing_points)

    def forward(self, x, *args, **kwargs):
        features = self.feature_extractor(x=x, *args, **kwargs)
        return self.gp_layer(features.squeeze(-1))


class SVKDL(BaseUncertainty):
    def __init__(
        self, base_model_class, hidden_size, num_inducing_points, *args, **kwargs
    ):
        super(SVKDL, self).__init__()
        self.base_model = base_model_class(out_features=hidden_size, *args, **kwargs)

        self.hidden_size = hidden_size
        self.num_inducing_points = num_inducing_points

        self.inducing_points = torch.tensor(
            np.random.randn(num_inducing_points, hidden_size), dtype=torch.float32
        ).requires_grad_(True)

        self.model = SVGPModel(self.inducing_points, self.base_model)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        self.train_losses_energy = []
        self.train_losses_force = []
        self.train_losses_total = []
        self.train_time = 0

        self.valid_losses_energy = []
        self.valid_losses_force = []
        self.valid_losses_total = []
        self.valid_time = 0

        self.test_losses_energy = []
        self.test_losses_force = []
        self.test_losses_total = []
        self.test_time = 0

    def fit(
        self,
        epochs,
        train_loader,
        valid_loader,
        device,
        dtype,
        model_path="gnn/models/evidential.pt",
        use_wandb=False,
        warmup_steps=0,
        force_weight=1.0,
        energy_weight=1.0,
        log_interval=100,
        patience=200,
        factor=0.1,
        lr=1e-3,
        min_lr=1e-6,
        additional_logs=None,
        best_on_train=False,
        coeff=None,
        test_loader=None,
    ):

        optimizer = torch.optim.Adam(
            [
                {"params": self.model.feature_extractor.parameters(), "lr": lr},
                {"params": self.model.gp_layer.parameters(), "lr": lr},
                {"params": self.likelihood.parameters(), "lr": lr},
            ],
            lr=lr,
            weight_decay=1e-16,
        )
        criterion = gpytorch.mlls.VariationalELBO(
            self.likelihood,
            self.model.gp_layer,
            num_data=train_loader.dataset.__len__(),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=2001, gamma=1 / (np.sqrt(10))
        )

        if use_wandb:
            self.init_wandb(
                scheduler,
                criterion,
                optimizer,
                model_path,
                train_loader,
                valid_loader,
                epochs,
                lr,
                patience,
                factor,
                force_weight,
                energy_weight,
            )

        best_valid_loss = np.inf

        for epoch in range(epochs):
            self.train_epoch(
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                epoch=epoch,
                device=device,
                dtype=dtype,
                force_weight=force_weight,
                energy_weight=energy_weight,
                log_interval=log_interval,
            )
            self.valid_epoch(
                valid_loader=valid_loader,
                criterion=criterion,
                device=device,
                dtype=dtype,
                force_weight=force_weight,
                energy_weight=energy_weight,
            )
            if test_loader is not None:
                self.valid_epoch(
                    valid_loader=test_loader,
                    criterion=criterion,
                    device=device,
                    dtype=dtype,
                    force_weight=force_weight,
                    energy_weight=energy_weight,
                    test=True,
                )

            self.epoch_summary(
                epoch,
                use_wandb=use_wandb,
                lr=optimizer.param_groups[0]["lr"],
                additional_logs=additional_logs,
            )

            if best_on_train:
                if np.array(self.train_losses_total).mean() < best_valid_loss:
                    best_valid_loss = np.array(self.train_losses_total).mean()
                    if model_path is not None:
                        torch.save(self.state_dict(), model_path)
                    self.best_model = self.state_dict()
            else:
                if np.array(self.valid_losses_total).mean() < best_valid_loss:
                    best_valid_loss = np.array(self.valid_losses_total).mean()
                    if model_path is not None:
                        torch.save(self.state_dict(), model_path)
                    self.best_model = self.state_dict()

            self.lr_before = optimizer.param_groups[0]["lr"]
            # scheduler.step(np.array(self.valid_losses_total).mean())
            scheduler.step()
            self.lr_after = optimizer.param_groups[0]["lr"]
            self.drop_metrics()

        if use_wandb:
            wandb.finish()

    def train_epoch(
        self,
        train_loader,
        optimizer,
        criterion,
        epoch,
        device,
        dtype,
        force_weight=1.0,
        energy_weight=1.0,
        log_interval=100,
    ):
        start = time.time()
        self.train()
        self.model.feature_extractor.train()
        self.model.gp_layer.train()
        self.likelihood.train()
        force_criterion = nn.L1Loss()
        for i, data in enumerate(train_loader):

            (
                atom_positions,
                nodes,
                edges,
                atom_mask,
                edge_mask,
                label_energy,
                label_forces,
                n_nodes,
            ) = self.prepare_data(data, device, dtype)

            energy, force, variance, output = self.forward(
                x=atom_positions,
                h0=nodes,
                edges=edges,
                edge_attr=None,
                node_mask=atom_mask,
                edge_mask=edge_mask,
                n_nodes=n_nodes,
            )

            loss_energy = -criterion(output, label_energy.unsqueeze(0))
            l1_energy = force_criterion(energy, label_energy)
            loss_force = force_criterion(force, label_forces)
            total_loss = force_weight * loss_force + energy_weight * loss_energy

            optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0)
            optimizer.step()

            self.train_losses_energy.append(
                l1_energy.item() * train_loader.dataset.std_energy
            )
            self.train_losses_force.append(
                loss_force.item() * train_loader.dataset.std_energy
            )
            self.train_losses_total.append(total_loss.item())

            if (i + 1) % log_interval == 0:
                print(
                    f"Epoch {epoch}, Batch {i+1}/{len(train_loader)}, Loss: {total_loss.mean().item()}, Variance: {torch.mean(variance)}",
                    flush=True,
                )

        self.train_time = time.time() - start

    def valid_epoch(
        self,
        valid_loader,
        criterion,
        device,
        dtype,
        force_weight=1.0,
        energy_weight=1.0,
        test=False,
    ):
        start = time.time()
        self.eval()
        self.model.feature_extractor.eval()
        self.model.gp_layer.eval()
        self.likelihood.eval()
        force_criterion = nn.L1Loss()
        for i, data in enumerate(valid_loader):
            (
                atom_positions,
                nodes,
                edges,
                atom_mask,
                edge_mask,
                label_energy,
                label_forces,
                n_nodes,
            ) = self.prepare_data(data, device, dtype)

            energy, force, variance, output = self.forward(
                x=atom_positions,
                h0=nodes,
                edges=edges,
                edge_attr=None,
                node_mask=atom_mask,
                edge_mask=edge_mask,
                n_nodes=n_nodes,
            )

            loss_energy = -criterion(output, label_energy)
            l1_energy = force_criterion(energy, label_energy)
            loss_force = force_criterion(force, label_forces)
            total_loss = force_weight * loss_force + energy_weight * loss_energy

            if not test:
                self.valid_losses_energy.append(
                    l1_energy.item() * valid_loader.dataset.std_energy
                )
                self.valid_losses_force.append(
                    loss_force.item() * valid_loader.dataset.std_energy
                )
                self.valid_losses_total.append(total_loss.item())
            else:
                self.test_losses_energy.append(
                    loss_energy.item() * valid_loader.dataset.std_energy
                )
                self.test_losses_force.append(
                    loss_force.item() * valid_loader.dataset.std_energy
                )
                self.test_losses_total.append(total_loss.item())
        if not test:
            self.valid_time = time.time() - start
        else:
            self.test_time = time.time() - start

    def predict(self, x, *args, use_force_uncertainty=False, **kwargs):
        self.eval()
        energy, force, variance, _ = self.forward(x=x, *args, **kwargs)
        uncertainty = variance.sqrt()
        return (
            energy,
            force,
            uncertainty * self.uncertainty_slope + self.uncertainty_bias,
        )

    def forward(self, x, *args, **kwargs):
        output = self.likelihood(self.model.forward(x=x, *args, **kwargs))
        energy = output.rsample()
        variance = output.variance

        grad_output = torch.ones_like(energy)
        force = -torch.autograd.grad(
            outputs=energy, inputs=x, grad_outputs=grad_output, create_graph=True
        )[0]
        return energy, force, variance, output

    def drop_metrics(self):
        self.train_losses_energy = []
        self.train_losses_force = []
        self.train_losses_total = []
        self.train_time = 0

        self.valid_losses_energy = []
        self.valid_losses_force = []
        self.valid_losses_total = []
        self.valid_time = 0

        self.test_losses_energy = []
        self.test_losses_force = []
        self.test_losses_total = []
        self.test_time = 0

    def epoch_summary(self, epoch, additional_logs=None, use_wandb=False, lr=None):
        attributes = [
            "train_losses_energy",
            "train_losses_force",
            "train_losses_total",
            "valid_losses_energy",
            "valid_losses_force",
            "valid_losses_total",
            "test_losses_energy",
            "test_losses_force",
            "test_losses_total",
        ]

        for attr in attributes:
            if getattr(self, attr) == []:
                setattr(self, attr, [0])

        print("", flush=True)
        print(f"Training and Validation Results of Epoch {epoch}:", flush=True)
        print("================================")
        print(
            f"Training Loss Energy: {np.array(self.train_losses_energy).mean()}, Training Loss Force: {np.array(self.train_losses_force).mean()}, time: {self.train_time}",
            flush=True,
        )
        if len(self.valid_losses_energy) > 0:
            print(
                f"Validation Loss Energy: {np.array(self.valid_losses_energy).mean()}, Validation Loss Force: {np.array(self.valid_losses_force).mean()}, time: {self.valid_time}",
                flush=True,
            )
        if len(self.test_losses_energy) > 0:
            print(
                f"Test Loss Energy: {np.array(self.test_losses_energy).mean()}, Test Loss Force: {np.array(self.test_losses_force).mean()}, time: {self.test_time}",
                flush=True,
            )
        print("", flush=True)

        logs = {
            "train_error_energy": np.array(self.train_losses_energy).mean(),
            "train_error_force": np.array(self.train_losses_force).mean(),
            "train_loss": np.array(self.train_losses_total).mean(),
            "valid_error_energy": np.array(self.valid_losses_energy).mean(),
            "valid_error_force": np.array(self.valid_losses_force).mean(),
            "valid_loss": np.array(self.valid_losses_total).mean(),
            "test_error_energy": np.array(self.test_losses_energy).mean(),
            "test_error_force": np.array(self.test_losses_force).mean(),
            "test_loss": np.array(self.test_losses_total).mean(),
            "lr": lr,
        }
        if additional_logs is not None:
            logs.update(additional_logs)

        if use_wandb:
            wandb.log(logs)

    def init_wandb(
        self,
        scheduler,
        criterion,
        optimizer,
        model_path,
        train_loader,
        valid_loader,
        epochs,
        lr,
        patience,
        factor,
        force_weight,
        energy_weight,
    ):
        wandb.init(
            # set the wandb project where this run will be logged
            project="GNN-Uncertainty-SVKDL",
            name=self.wandb_name,
            # track hyperparameters and run metadata
            config={
                "name": "alaninedipeptide",
                "learning_rate_start": lr,
                "layers": self.model.feature_extractor.n_layers,
                "hidden_nf": self.model.feature_extractor.hidden_nf,
                "scheduler": type(scheduler).__name__,
                "optimizer": type(optimizer).__name__,
                "patience": patience,
                "factor": factor,
                "dataset": len(train_loader.dataset) + len(valid_loader.dataset),
                "epochs": epochs,
                "batch_size": train_loader.batch_size,
                "in_node_nf": self.model.feature_extractor.in_node_nf,
                "in_edge_nf": self.model.feature_extractor.in_edge_nf,
                "loss_fn": type(criterion).__name__,
                "model_checkpoint": model_path,
                "force_weight": force_weight,
                "energy_weight": energy_weight,
            },
        )
