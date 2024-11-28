import torch
from torch import nn

from abc import abstractmethod

import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    HuberRegressor,
    RANSACRegressor,
)
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from scipy.odr import ODR, Model, RealData


from datasets.helper import utils as qm9_utils
import csv
import os
import itertools


class BaseUncertainty(nn.Module):

    @abstractmethod
    def predict(self, x, *args, **kwargs):
        pass

    @abstractmethod
    def fit(
        self,
        epochs,
        train_loader,
        valid_loader,
        device,
        dtype,
        model_path,
        use_wandb=False,
        force_weight=1.0,
        energy_weight=1.0,
        log_interval=100,
        patience=200,
        factor=0.1,
        lr=1e-3,
        min_lr=1e-6,
    ):
        pass

    def __init__(self):
        super(BaseUncertainty, self).__init__()
        self.best_model = self.state_dict()
        self.uncertainty_slope = 1.0
        self.uncertainty_bias = 0.0
        self.wandb_name = None

    def prepare_data(self, data, device, dtype):
        batch_size, n_nodes, _ = data["coordinates"].size()
        atom_positions = (
            data["coordinates"]
            .view(batch_size * n_nodes, -1)
            .requires_grad_(True)
            .to(device, dtype)
        )
        atom_mask = data["atom_mask"].view(batch_size * n_nodes, -1).to(device, dtype)
        edge_mask = (
            data["edge_mask"].view(batch_size * n_nodes * n_nodes, -1).to(device, dtype)
        )
        one_hot = data["one_hot"].to(device, dtype)
        charges = data["charges"].to(device, dtype)
        charge_scale = data["charge_scale"][0]
        charge_power = data["charge_power"][0]

        nodes = qm9_utils.preprocess_input(
            one_hot, charges, charge_power, charge_scale, device
        )

        nodes = nodes.view(batch_size * n_nodes, -1)

        edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, device)
        label_energy = (data["energies"]).to(device, dtype)
        label_forces = (data["forces"]).view(batch_size * n_nodes, -1).to(device, dtype)

        return (
            atom_positions,
            nodes,
            edges,
            atom_mask,
            edge_mask,
            label_energy,
            label_forces,
            n_nodes,
        )

    def evaluate_uncertainty(
        self, test_loader, device, dtype, plot_path=None, csv_path=None, show_plot=False
    ):
        criterion = nn.L1Loss(reduction="none")
        energy_losses = torch.Tensor()
        uncertainties = torch.Tensor()
        self.eval()
        for i, data in enumerate(test_loader):
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

            energy, forces, uncertainty = self.predict(
                x=atom_positions,
                h0=nodes,
                edges=edges,
                edge_attr=None,
                node_mask=atom_mask,
                edge_mask=edge_mask,
                n_nodes=n_nodes,
            )
            energy_losses = torch.cat(
                (energy_losses, criterion(energy.cpu().detach(), label_energy.cpu())),
                dim=0,
            )
            uncertainties = torch.cat(
                (uncertainties, uncertainty.cpu().detach()), dim=0
            )

            atom_positions.detach()

        energy_losses = energy_losses.cpu().detach().numpy()
        uncertainties = uncertainties.cpu().detach().numpy()

        correlation = np.corrcoef(energy_losses, uncertainties)[0, 1]
        self._scatter_plot(
            energy_losses,
            uncertainties,
            self.__class__.__name__,
            "Energy Losses",
            "Uncertainties",
            text=f"Correlation: {correlation}",
            save_path=plot_path,
            show_plot=show_plot,
        )
        print(f"Correlation: {correlation}")
        if csv_path:
            if os.path.exists(csv_path):
                with open(csv_path, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([self.__class__.__name__, correlation])
            else:
                with open(csv_path, "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["Method", "Correlation Coefficient"])
                    writer.writerow([self.__class__.__name__, correlation])

    def evaluate_model(
        self, test_loader, device, dtype, plot_path=None, csv_path=None, show_plot=False
    ):
        criterion = nn.L1Loss(reduction="none")
        predictions_energy = torch.Tensor()
        ground_truths_energy = torch.Tensor()
        self.eval()
        for i, data in enumerate(test_loader):
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

            energy, forces, uncertainty = self.predict(
                x=atom_positions,
                h0=nodes,
                edges=edges,
                edge_attr=None,
                node_mask=atom_mask,
                edge_mask=edge_mask,
                n_nodes=n_nodes,
            )
            predictions_energy = torch.cat(
                (predictions_energy, energy.cpu().detach()), dim=0
            )
            ground_truths_energy = torch.cat(
                (ground_truths_energy, label_energy.cpu().detach()), dim=0
            )

            atom_positions.detach()

        ground_truths_energy = ground_truths_energy.cpu().detach().numpy()
        predictions_energy = predictions_energy.cpu().detach().numpy()

        energy_r2 = r2_score(ground_truths_energy, predictions_energy)

        self._scatter_plot(
            ground_truths_energy,
            predictions_energy,
            self.__class__.__name__,
            "Ground Truth Energy",
            "Predicted Energy",
            text=f"Energy R2 Score: {energy_r2}",
            save_path=plot_path,
            show_plot=show_plot,
        )
        if csv_path:
            if os.path.exists(csv_path):
                with open(csv_path, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([self.__class__.__name__, energy_r2])
            else:
                with open(csv_path, "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["Method", "Energy R2 Score"])
                    writer.writerow([self.__class__.__name__, energy_r2])

    def evaluate_all(
        self,
        test_loader_in,
        device,
        dtype,
        test_loader_out=None,
        plot_name=None,
        csv_path=None,
        show_plot=None,
        best_model_available=True,
        use_energy_uncertainty=False,
        use_force_uncertainty=True,
    ):
        if best_model_available:
            self.load_state_dict(self.best_model)
        scaling = test_loader_in.dataset.std_energy

        if use_energy_uncertainty:
            (
                energy_r2_in,
                forces_r2_in,
                energy_correlation_in_energy,
                energy_correlation_in_force,
                energy_losses_in,
                force_losses_in,
                uncertainties_in,
            ) = self._eval_all(
                test_loader_in,
                device,
                dtype,
                plot_path=f"{plot_name}_in",
                plot_title=self.__class__.__name__ + " In Distribution",
                use_force_uncertainty=False,
                plot_loss=True,
                plot=not (test_loader_out is not None),
            )
            print(np.min(force_losses_in), np.min(energy_losses_in))
            print(np.min(uncertainties_in), np.max(uncertainties_in))
            if test_loader_out is not None:
                (
                    energy_r2_out,
                    forces_r2_out,
                    energy_correlation_out_energy,
                    energy_correlation_out_forces,
                    energy_losses_out,
                    force_losses_out,
                    uncertainties_out,
                ) = self._eval_all(
                    test_loader_out,
                    device,
                    dtype,
                    plot_path=f"{plot_name}_out",
                    plot_title=self.__class__.__name__ + " Out Distribution",
                    use_force_uncertainty=False,
                    plot_loss=True,
                    plot=False,
                )
                self._mulit_scatter_plot(
                    uncertainties_in,
                    energy_losses_in,
                    uncertainties_out,
                    energy_losses_out,
                    self.__class__.__name__,
                    "Energy Uncertainties",
                    "Energy Errors",
                    text=f"Correlation in: {energy_correlation_in_energy}\nCorrelation Out: {energy_correlation_out_energy}",
                    save_path=plot_name + "_energy_uncertainty_energy_loss.svg",
                    show_plot=show_plot,
                )
                self._mulit_scatter_plot(
                    uncertainties_in,
                    np.mean(
                        force_losses_in.reshape(energy_losses_in.shape[0], -1, 3),
                        axis=(1, 2),
                    ),
                    uncertainties_out,
                    np.mean(
                        force_losses_out.reshape(energy_losses_out.shape[0], -1, 3),
                        axis=(1, 2),
                    ),
                    self.__class__.__name__,
                    "Energy Uncertainties",
                    "Force Errors",
                    text=f"Correlation in: {energy_correlation_in_force}\nCorrelation Out: {energy_correlation_out_forces}",
                    save_path=plot_name + "_energy_uncertainty_force_loss.svg",
                    show_plot=show_plot,
                )

        if use_force_uncertainty:
            (
                energy_r2_in,
                forces_r2_in,
                force_correlation_in_energy,
                force_correlation_in_forces,
                energy_losses_in,
                force_losses_in,
                uncertainties_in,
            ) = self._eval_all(
                test_loader_in,
                device,
                dtype,
                plot_path=f"{plot_name}_in",
                plot_title=self.__class__.__name__ + " In Distribution",
                use_force_uncertainty=True,
                plot_loss=not use_energy_uncertainty,
                plot=not (test_loader_out is not None),
            )
            if test_loader_out is not None:
                (
                    energy_r2_out,
                    forces_r2_out,
                    force_correlation_out_energy,
                    force_correlation_out_forces,
                    energy_losses_out,
                    force_losses_out,
                    uncertainties_out,
                ) = self._eval_all(
                    test_loader_out,
                    device,
                    dtype,
                    plot_path=f"{plot_name}_out",
                    plot_title=self.__class__.__name__ + " Out Distribution",
                    use_force_uncertainty=True,
                    plot=False,
                )
                self._mulit_scatter_plot(
                    uncertainties_in,
                    energy_losses_in,
                    uncertainties_out,
                    energy_losses_out,
                    self.__class__.__name__,
                    "Force Uncertainties",
                    "Energy Errors",
                    text=f"Correlation in: {force_correlation_in_energy}\nCorrelation Out: {force_correlation_out_energy}",
                    save_path=plot_name + "_force_uncertainty_energy_loss.svg",
                    show_plot=show_plot,
                )
                self._mulit_scatter_plot(
                    uncertainties_in,
                    np.mean(
                        force_losses_in.reshape(energy_losses_in.shape[0], -1, 3),
                        axis=(1, 2),
                    ),
                    uncertainties_out,
                    np.mean(
                        force_losses_out.reshape(energy_losses_out.shape[0], -1, 3),
                        axis=(1, 2),
                    ),
                    self.__class__.__name__,
                    "Force Uncertainties",
                    "Force Errors",
                    text=f"Correlation in: {force_correlation_in_forces}\nCorrelation Out: {force_correlation_out_forces}",
                    save_path=plot_name + "_force_uncertainty_force_loss.svg",
                    show_plot=show_plot,
                )

        if not use_energy_uncertainty:
            energy_correlation_out_energy, energy_correlation_out_forces = 0, 0
        if not use_force_uncertainty:
            force_correlation_out_energy, force_correlation_out_forces = 0, 0
        if test_loader_out is None:
            (
                energy_r2_out,
                forces_r2_out,
                energy_correlation_out_energy,
                energy_correlation_out_forces,
                energy_losses_out,
                force_losses_out,
            ) = (0, 0, 0, 0, np.array([0]), np.array([[[0]]]))

        if csv_path:
            row = [
                self.__class__.__name__,
                energy_r2_in,
                forces_r2_in,
                np.mean(energy_losses_in),
                np.mean(force_losses_in),
                energy_r2_out,
                forces_r2_out,
                np.mean(energy_losses_out),
                np.mean(force_losses_out),
            ]
            if use_energy_uncertainty:
                row.extend(
                    [
                        energy_correlation_in_energy,
                        energy_correlation_in_force,
                        energy_correlation_out_energy,
                        energy_correlation_out_forces,
                    ]
                )
            if use_force_uncertainty:
                row.extend(
                    [
                        force_correlation_in_energy,
                        force_correlation_in_forces,
                        force_correlation_out_energy,
                        force_correlation_out_forces,
                    ]
                )
            if os.path.exists(csv_path):
                with open(csv_path, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(row)
            else:
                with open(csv_path, "w", newline="") as file:
                    writer = csv.writer(file)
                    titles = [
                        "Method",
                        "Energy R2 Score In Distribution",
                        "Forces R2 Score In Distribution",
                        "Energy Errors In Distribution",
                        "Force Errors In Distribution",
                        "Energy R2 Score Out Distribution",
                        "Force R2 Score Out Distribution",
                        "Energy Errors Out Distribution",
                        "Force Errors Out Distribution",
                    ]
                    if use_energy_uncertainty:
                        titles.extend(
                            [
                                "Energy Correlation In Distribution Energies",
                                "Energy Correlation In Distribution Forces",
                                "Energy Correlation Out Distribution Energy",
                                "Energy Correlation Out Distribution Forces",
                            ]
                        )
                    if use_force_uncertainty:
                        titles.extend(
                            [
                                "Force Correlation In Distribution Energy",
                                "Force Correlation In Distribution Forces",
                                "Force Correlation Out Distribution Energy",
                                "Force Correlation Out Distribution Forces",
                            ]
                        )
                    writer.writerow(titles)
                    writer.writerow(row)

    def _eval_all(
        self,
        dataloader,
        device,
        dtype,
        plot_path=None,
        plot_title=None,
        use_force_uncertainty=False,
        plot_loss=False,
        plot=True,
    ):
        criterion = nn.L1Loss(reduction="none")
        predictions_energy = torch.Tensor()
        ground_truths_energy = torch.Tensor()
        predictions_forces = torch.Tensor()
        ground_truths_forces = torch.Tensor()
        energy_losses = torch.Tensor()
        forces_losses = torch.Tensor()
        uncertainties = torch.Tensor()

        scaling = dataloader.dataset.std_energy

        self.eval()
        for i, data in enumerate(dataloader):
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
            if use_force_uncertainty:
                energy, forces, uncertainty = self.predict(
                    x=atom_positions,
                    h0=nodes,
                    edges=edges,
                    edge_attr=None,
                    node_mask=atom_mask,
                    edge_mask=edge_mask,
                    n_nodes=n_nodes,
                    use_force_uncertainty=True,
                )
            else:
                energy, forces, uncertainty = self.predict(
                    x=atom_positions,
                    h0=nodes,
                    edges=edges,
                    edge_attr=None,
                    node_mask=atom_mask,
                    edge_mask=edge_mask,
                    n_nodes=n_nodes,
                    use_force_uncertainty=False,
                )

            energy = energy * scaling
            label_energy = label_energy * scaling
            forces = forces * scaling
            label_forces = label_forces * scaling
            uncertainty = uncertainty * scaling

            predictions_energy = torch.cat(
                (predictions_energy, energy.cpu().detach()), dim=0
            )
            ground_truths_energy = torch.cat(
                (ground_truths_energy, label_energy.cpu().detach()), dim=0
            )
            predictions_forces = torch.cat(
                (predictions_forces, forces.cpu().detach()), dim=0
            )
            ground_truths_forces = torch.cat(
                (ground_truths_forces, label_forces.cpu().detach()), dim=0
            )
            energy_losses = torch.cat(
                (energy_losses, criterion(energy.cpu().detach(), label_energy.cpu())),
                dim=0,
            )
            forces_losses = torch.cat(
                (forces_losses, criterion(forces.cpu().detach(), label_forces.cpu())),
                dim=0,
            )
            uncertainties = torch.cat(
                (uncertainties, uncertainty.cpu().detach()), dim=0
            )

            atom_positions.detach()
        ground_truths_energy = ground_truths_energy.cpu().detach().numpy()
        predictions_energy = predictions_energy.cpu().detach().numpy()
        ground_truths_forces = ground_truths_forces.cpu().detach().numpy()
        predictions_forces = predictions_forces.cpu().detach().numpy()
        energy_losses = energy_losses.cpu().detach().numpy()
        forces_losses = forces_losses.cpu().detach().numpy()
        uncertainties = uncertainties.cpu().detach().numpy()
        energy_r2 = r2_score(ground_truths_energy, predictions_energy)
        forces_r2 = r2_score(ground_truths_forces, predictions_forces)
        correlation_energy = np.corrcoef(energy_losses, uncertainties)[0, 1]
        correlation_forces = np.corrcoef(
            np.mean(forces_losses.reshape(energy_losses.shape[0], -1, 3), axis=(1, 2)),
            uncertainties,
        )[0, 1]

        if plot_path:
            if plot_loss:
                self._scatter_plot(
                    ground_truths_energy,
                    predictions_energy,
                    plot_title,
                    "Ground Truth Energy",
                    "Predicted Energy",
                    text=f"Energy R2 Score: {energy_r2}",
                    save_path=plot_path + "_energy.svg",
                    show_plot=False,
                )
            uncertainty_type = "_force" if use_force_uncertainty else "_energy"
            if plot:
                self._scatter_plot(
                    energy_losses,
                    uncertainties,
                    plot_title,
                    "Energy Errors",
                    "Uncertainties",
                    text=f"Correlation: {correlation_energy}",
                    save_path=plot_path
                    + uncertainty_type
                    + "_uncertainty_energy_loss.svg",
                    show_plot=False,
                )
                self._scatter_plot(
                    np.mean(
                        forces_losses.reshape(energy_losses.shape[0], -1, 3),
                        axis=(1, 2),
                    ),
                    uncertainties,
                    plot_title,
                    "Force Errors",
                    "Uncertainties",
                    text=f"Correlation: {correlation_forces}",
                    save_path=plot_path
                    + uncertainty_type
                    + "_uncertainty_force_loss.svg",
                    show_plot=False,
                )

        return (
            energy_r2,
            forces_r2,
            correlation_energy,
            correlation_forces,
            energy_losses,
            forces_losses,
            uncertainties,
        )

    def _scatter_plot(
        self, x, y, title, xlabel, ylabel, text="", save_path=None, show_plot=False
    ):
        # plt.scatter(x, y)
        plt.figure(figsize=(10, 8))
        sns.kdeplot(x=x, y=y, cmap="Blues", fill=True)  # Density plot
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.plot([min(x), max(x)], [min(x), max(x)], color="red", linestyle="--")
        plt.text(0.1, 0.9, text, transform=plt.gca().transAxes)
        if save_path is not None:
            plt.savefig(save_path)
        if show_plot:
            plt.show()
        plt.close()

    def _mulit_scatter_plot(
        self,
        x_in,
        y_in,
        x_out,
        y_out,
        title,
        xlabel,
        ylabel,
        text="",
        save_path=None,
        show_plot=False,
    ):

        plt.figure(figsize=(10, 8))

        # Define a color cycle if colors aren't specified in the data_list
        colors = itertools.cycle(sns.color_palette("hsv", 2))

        sns.kdeplot(
            x=x_in,
            y=y_in,
            cmap=sns.light_palette("blue", as_cmap=True),
            fill=True,
            label="In Distribution",
            alpha=1,
            legend=True,
        )

        sns.kdeplot(
            x=x_out,
            y=y_out,
            cmap=sns.light_palette("green", as_cmap=True),
            fill=True,
            label="Out of Distribution",
            alpha=0.5,
            legend=True,
        )
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        # Add a diagonal line across the plot
        min_val, max_val = np.percentile(
            np.concatenate((x_in, y_in, x_out, y_out)), [0, 99.9]
        )
        # max_val = max(max(x_in), max(y_in), max(x_out), max(y_out))
        # min_val = min(min(x_in), min(y_in), min(x_out), min(y_out))
        plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")

        red_patch = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markersize=5,
            label="In Distribution",
            linestyle="None",
        )
        blue_patch = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="green",
            markersize=5,
            label="Out of Distribution",
            linestyle="None",
        )
        plt.legend(handles=[red_patch, blue_patch], loc="upper right")

        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)

        plt.text(0.1, 0.9, text, transform=plt.gca().transAxes)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)
        if show_plot:
            plt.show()
        plt.close()

    def calibrate_uncertainty(self, validation_loader, device, dtype, path=None):
        criterion = nn.L1Loss(reduction="none")
        force_losses = torch.Tensor()
        uncertainties = torch.Tensor()
        scaling = validation_loader.dataset.std_energy
        self.eval()
        self.uncertainty_slope = 1.0
        self.uncertainty_bias = 0.0
        for i, data in enumerate(validation_loader):
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

            energy, forces, uncertainty = self.predict(
                x=atom_positions,
                h0=nodes,
                edges=edges,
                edge_attr=None,
                node_mask=atom_mask,
                edge_mask=edge_mask,
                n_nodes=n_nodes,
            )
            forces = forces * scaling
            label_forces = label_forces * scaling
            uncertainty = uncertainty * scaling
            energy = energy * scaling
            force_losses = torch.cat(
                (force_losses, criterion(forces.cpu().detach(), label_forces.cpu())),
                dim=0,
            )
            uncertainties = torch.cat(
                (uncertainties, uncertainty.cpu().detach()), dim=0
            )

            atom_positions.detach()

        force_losses = force_losses.cpu().detach().numpy()
        force_losses = np.mean(
            force_losses.reshape(uncertainties.shape[0], -1, 3), axis=(1, 2)
        )
        uncertainties = uncertainties.cpu().detach().numpy()

        def linear_model(B, x):
            """Linear model y = B[0] * x + B[1]"""
            return B[0] * x + B[1]

        data = RealData(uncertainties / scaling, force_losses / scaling)
        model = Model(linear_model)
        linear = ODR(data, model, beta0=[1.0, 1.0])
        # Reshape the data for linear regression
        uncertainties = uncertainties.reshape(-1, 1)
        force_losses = force_losses.reshape(-1, 1)

        output = linear.run()
        slope, bias = output.beta

        # Create and fit the model
        linear = LinearRegression()
        linear.fit(uncertainties / scaling, force_losses / scaling)

        # Get the slope and intercept
        slope = linear.coef_[0][0]  # Coefficient for the feature
        bias = linear.intercept_[0]  # Intercept (bias term)
        self.uncertainty_slope = slope  # linear.coef_[0][0]
        self.uncertainty_bias = bias  # linear.intercept_[0]
        # self.uncertainty_slope = linear.coef_[0]
        # self.uncertainty_bias = linear.intercept_

        print(
            f"Uncertainty Slope: {self.uncertainty_slope}, Uncertainty Bias: {self.uncertainty_bias}",
            flush=True,
        )

        if path is not None:
            plt.figure(figsize=(10, 8))
            # sns.kdeplot(x=uncertainties, y=force_losses, cmap=sns.light_palette("blue", as_cmap=True), fill=True)
            plt.scatter(uncertainties / scaling, force_losses / scaling, label="Data")
            plt.plot(
                uncertainties,
                self.uncertainty_slope * uncertainties + self.uncertainty_bias,
                color="red",
                label="Regression Line",
            )
            plt.xlabel("Uncertainties")
            plt.ylabel("Force Errors")
            plt.title("Uncertainty Calibration")
            plt.legend()
            plt.tight_layout()
            plt.savefig(path)
            plt.close()

    def set_wandb_name(self, name):
        self.wandb_name = name

    def load_best_model(self):
        self.load_state_dict(self.best_model)

    def valid_on_cv(
        self, validationloader, save_path, device, dtype, use_force_uncertainty
    ):
        data = self._calc_all_np(validationloader, device, dtype, use_force_uncertainty)
        (
            predictions_energy,
            ground_truths_energy,
            predictions_forces,
            ground_truths_forces,
            energy_losses,
            forces_losses,
            uncertainties,
        ) = data
        coords = self._get_coords_from_dataloader(validationloader, device, dtype)
        print(coords.shape)

        indices_psi = [4, 6, 8, 14]
        indices_phi = [6, 8, 14, 16]

        psi = self._calculate_dihedrals_batch(coords, indices_psi)
        phi = self._calculate_dihedrals_batch(coords, indices_phi)

        print(psi.shape)

        self._heatmap(
            psi,
            -phi,
            energy_losses,
            "Energy",
            "φ",
            "Ψ",
            save_path + "_energy_error.svg",
            "Energy Error",
            num_bins=200,
        )
        self._heatmap(
            psi,
            -phi,
            uncertainties,
            "Uncertainties",
            "φ",
            "Ψ",
            save_path + "_uncertainties.svg",
            "Uncertainty",
            num_bins=200,
        )

    def _heatmap(
        self,
        x,
        y,
        val,
        title,
        x_label,
        y_label,
        save_path,
        data_label=None,
        num_bins=100,
    ):
        # Define bin size
        bin_size = 360 / (num_bins)

        # Create bins
        psi_bins = np.arange(-180, 180, bin_size)
        phi_bins = np.arange(-180, 180, bin_size)

        # Digitize psi and phi into bins
        psi_indices = np.digitize(x, bins=psi_bins) - 1  # Bin indices for psi
        phi_indices = np.digitize(y, bins=phi_bins) - 1  # Bin indices for phi

        # Create a grid as a list of lists to store values
        grid = [[[] for _ in range(len(psi_bins))] for _ in range(len(phi_bins))]

        # Populate the grid with uncertainties
        for i in range(len(x)):
            psi_idx = psi_indices[i]
            phi_idx = phi_indices[i]
            grid[phi_idx][psi_idx].append(val[i])

        # Calculate the average uncertainty for each bin
        average_grid = np.full((len(phi_bins) - 1, len(psi_bins) - 1), np.nan)

        for i in range(len(phi_bins) - 1):
            for j in range(len(psi_bins) - 1):
                if grid[i][j]:  # Check if the bin is not empty
                    average_grid[i, j] = np.mean(grid[i][j])

        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(
            average_grid,
            aspect="auto",
            cmap="viridis",
            extent=(psi_bins[0], psi_bins[-1], phi_bins[0], phi_bins[-1]),
            norm=LogNorm(vmin=1e-1, vmax=10),
        )
        if data_label is not None:
            plt.colorbar(label=data_label)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.savefig(save_path)

    def _calc_all(self, dataloader, device, dtype, use_force_uncertainty=None):
        self.eval()
        criterion = nn.L1Loss(reduction="none")

        predictions_energy = torch.Tensor()
        ground_truths_energy = torch.Tensor()
        predictions_forces = torch.Tensor()
        ground_truths_forces = torch.Tensor()
        energy_losses = torch.Tensor()
        forces_losses = torch.Tensor()
        uncertainties = torch.Tensor()

        for i, data in enumerate(dataloader):
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
            if use_force_uncertainty is not None:
                energy, forces, uncertainty = self.predict(
                    x=atom_positions,
                    h0=nodes,
                    edges=edges,
                    edge_attr=None,
                    node_mask=atom_mask,
                    edge_mask=edge_mask,
                    n_nodes=n_nodes,
                    use_force_uncertainty=use_force_uncertainty,
                )
            else:
                energy, forces, uncertainty = self.predict(
                    x=atom_positions,
                    h0=nodes,
                    edges=edges,
                    edge_attr=None,
                    node_mask=atom_mask,
                    edge_mask=edge_mask,
                    n_nodes=n_nodes,
                )

            predictions_energy = torch.cat(
                (predictions_energy, energy.cpu().detach()), dim=0
            )
            ground_truths_energy = torch.cat(
                (ground_truths_energy, label_energy.cpu().detach()), dim=0
            )
            predictions_forces = torch.cat(
                (predictions_forces, forces.cpu().detach()), dim=0
            )
            ground_truths_forces = torch.cat(
                (ground_truths_forces, label_forces.cpu().detach()), dim=0
            )
            energy_losses = torch.cat(
                (energy_losses, criterion(energy.cpu().detach(), label_energy.cpu())),
                dim=0,
            )
            forces_losses = torch.cat(
                (forces_losses, criterion(forces.cpu().detach(), label_forces.cpu())),
                dim=0,
            )
            uncertainties = torch.cat(
                (uncertainties, uncertainty.cpu().detach()), dim=0
            )
        return (
            predictions_energy,
            ground_truths_energy,
            predictions_forces,
            ground_truths_forces,
            energy_losses,
            forces_losses,
            uncertainties,
        )

    def _calc_all_np(self, dataloader, device, dtype, use_force_uncertainty=None):
        (
            predictions_energy,
            ground_truths_energy,
            predictions_forces,
            ground_truths_forces,
            energy_losses,
            forces_losses,
            uncertainties,
        ) = self._calc_all(dataloader, device, dtype, use_force_uncertainty)
        ground_truths_energy = ground_truths_energy.cpu().detach().numpy()
        predictions_energy = predictions_energy.cpu().detach().numpy()
        ground_truths_forces = ground_truths_forces.cpu().detach().numpy()
        predictions_forces = predictions_forces.cpu().detach().numpy()
        energy_losses = energy_losses.cpu().detach().numpy()
        forces_losses = forces_losses.cpu().detach().numpy()
        uncertainties = uncertainties.cpu().detach().numpy()
        return (
            predictions_energy,
            ground_truths_energy,
            predictions_forces,
            ground_truths_forces,
            energy_losses,
            forces_losses,
            uncertainties,
        )

    def _get_coords_from_dataloader(self, dataloader, device, dtype):
        coords = []
        for i, data in enumerate(dataloader):
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
            coords.append(atom_positions.cpu().detach().numpy().reshape(-1, n_nodes, 3))
        return np.concatenate(coords)

    def _calculate_dihedrals_batch(self, molecules, indices) -> np.array:
        """
        Calculate dihedral angles for a batch of molecules in a vectorized manner.

        Parameters:
        molecules (np.ndarray): Array of shape (n_molecules, n_atoms, 3) containing the coordinates of the atoms in each molecule.
        indices (list): List of 4 indices that define the dihedral angle for all molecules.

        Returns:
        np.ndarray: Array of dihedral angles for the batch of molecules.
        """
        n_molecules = molecules.shape[0]

        # Extract coordinates of the four atoms for all molecules
        p0 = molecules[:, indices[0], :]
        p1 = molecules[:, indices[1], :]
        p2 = molecules[:, indices[2], :]
        p3 = molecules[:, indices[3], :]

        # Compute the vectors between the points
        b0 = p1 - p0
        b1 = p2 - p1
        b2 = p3 - p2

        # Normalize b1
        b1 /= np.linalg.norm(b1, axis=1)[:, np.newaxis]

        # Compute normal vectors to the planes
        n0 = np.cross(b0, b1)
        n1 = np.cross(b1, b2)

        # Normalize the normal vectors
        n0 /= np.linalg.norm(n0, axis=1)[:, np.newaxis]
        n1 /= np.linalg.norm(n1, axis=1)[:, np.newaxis]

        # Compute the dihedral angles
        m1 = np.cross(n0, b1)
        x = np.einsum("ij,ij->i", n0, n1)
        y = np.einsum("ij,ij->i", m1, n1)
        angles = -np.degrees(np.arctan2(y, x))

        return angles

    def prepare_al_iteration(self):
        pass
