import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import BayesianGaussianMixture
import os


from matplotlib.animation import FuncAnimation, PillowWriter

plt.rcParams.update({"font.size": 22})


def calculate_dihedrals_batch(molecules, indices) -> np.array:
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


def plot_histogram(hist, xedges, yedges):
    plt.imshow(
        hist,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="viridis",
    )
    plt.colorbar()
    plt.xlabel("φ")
    plt.ylabel("Ψ")
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.title("Histogram of dihedral angles (log scale)")
    plt.tight_layout()
    plt.savefig("../../gnn/histogram_train.png")
    # plt.show()


def read_xyz(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        coordinates = []
        sum = 0
        molecules = []
        energies = []
        while True:
            if sum >= len(lines):
                break
            num_atoms = int(lines[sum])
            coordinates = []
            energies.append(float(lines[sum + 1]))
            for line in lines[sum + 2 : sum + num_atoms + 2]:
                atom, x, y, z, fx, fy, fz = line.split()
                coordinates.append([float(x), float(y), float(z)])
            sum += num_atoms + 2
            molecules.append(coordinates)
        molecules = np.array(molecules)
        energies = np.array(energies)
        return molecules, energies


def plot_multiple_histograms(
    hists: list, xedges: list, yedges: list, titles: list = None
):
    fig, axs = plt.subplots(1, len(hists), figsize=(15, 5))
    for i in range(len(hists)):
        hist = hists[i]
        xedge = xedges[i]
        yedge = yedges[i]
        axs[i].imshow(
            hist,
            origin="lower",
            extent=[xedge[0], xedge[-1], yedge[0], yedge[-1]],
            cmap="viridis",
        )
        axs[i].set_xlabel("φ")
        axs[i].set_ylabel("Ψ")
        if titles is not None:
            axs[i].set_title(titles[i])
    plt.tight_layout()
    plt.show()


def get_histogram_difference(hist1, hist2):

    return np.mean(np.abs(hist1 - hist2))


def create_single_hist(path):
    with open(path, "r") as f:
        lines = f.readlines()
        coordinates = []
        sum = 0
        molecules = []
        energies = []
        while True:
            if sum >= len(lines):
                break
            num_atoms = int(lines[sum])
            coordinates = []
            energies.append(float(lines[sum + 1]))
            for line in lines[sum + 2 : sum + num_atoms + 2]:
                atom, x, y, z, fx, fy, fz = line.split()
                coordinates.append([float(x), float(y), float(z)])
            sum += num_atoms + 2
            molecules.append(coordinates)
        molecules = np.array(molecules)
        energies = np.array(energies)
    print(len(molecules))

    indizes1 = [4, 6, 8, 14]
    indizes2 = [6, 8, 14, 16]

    psi = calculate_dihedrals_batch(molecules, indizes1)
    phi = calculate_dihedrals_batch(molecules, indizes2)

    hist, xedges, yedges = np.histogram2d(
        phi, psi, bins=500, range=[[-180, 180], [-180, 180]]
    )
    log_hist = hist  # np.log(hist+1)
    plot_histogram(log_hist, xedges, yedges)


def single_scatter(path):
    with open(path, "r") as f:
        lines = f.readlines()
        coordinates = []
        sum = 0
        molecules = []
        energies = []
        while True:
            if sum >= len(lines):
                break
            num_atoms = int(lines[sum])
            coordinates = []
            energies.append(float(lines[sum + 1]))
            for line in lines[sum + 2 : sum + num_atoms + 2]:
                atom, x, y, z, fx, fy, fz = line.split()
                coordinates.append([float(x), float(y), float(z)])
            sum += num_atoms + 2
            molecules.append(coordinates)
        molecules = np.array(molecules)
        energies = np.array(energies)
    print(len(molecules))

    indizes1 = [4, 6, 8, 14]
    indizes2 = [6, 8, 14, 16]

    psi = calculate_dihedrals_batch(molecules, indizes1)
    phi = calculate_dihedrals_batch(molecules, indizes2)

    plt.scatter(psi, phi, c=np.ones_like(psi), s=1)
    plt.ylim(-180, 180)
    plt.xlim(-180, 180)
    plt.xlabel("φ")
    plt.ylabel("Ψ")
    plt.show()


def multi_scatter(paths):
    colors = ["r", "g", "b", "c", "m", "y", "k"]
    plt.figure()

    for i, path in enumerate(paths):
        with open(path, "r") as f:
            lines = f.readlines()
            coordinates = []
            sum = 0
            molecules = []
            energies = []
            while True:
                if sum >= len(lines):
                    break
                num_atoms = int(lines[sum])
                coordinates = []
                energies.append(float(lines[sum + 1]))
                for line in lines[sum + 2 : sum + num_atoms + 2]:
                    atom, x, y, z, fx, fy, fz = line.split()
                    coordinates.append([float(x), float(y), float(z)])
                sum += num_atoms + 2
                molecules.append(coordinates)
            molecules = np.array(molecules)
            energies = np.array(energies)
        print(len(molecules))

        indizes1 = [4, 6, 8, 14]
        indizes2 = [6, 8, 14, 16]

        psi = calculate_dihedrals_batch(molecules, indizes1)
        phi = calculate_dihedrals_batch(molecules, indizes2)

        plt.scatter(psi, phi, c=colors[i % len(colors)], s=2, label=f"{path}", alpha=1)

    plt.ylim(-180, 180)
    plt.xlim(-180, 180)
    plt.xlabel("φ")
    plt.ylabel("Ψ")
    plt.legend()
    plt.show()


def animate_active_learning(base_path, paths, fps=15):
    plt.clf()

    fig, ax = plt.subplots(figsize=(8, 8))

    plt.ylim(-180, 180)
    plt.xlim(-180, 180)

    # Read initial data and calculate base dihedrals
    base_molecules, base_energies = read_xyz(base_path)
    indices1 = [4, 6, 8, 14]
    indices2 = [6, 8, 14, 16]
    base_psi = calculate_dihedrals_batch(base_molecules, indices1)
    base_phi = calculate_dihedrals_batch(base_molecules, indices2)

    # Initialize colors array: start with red for all base points
    colors = ["red"] * len(base_psi)

    # Create the initial scatter plot
    scatter = ax.scatter(base_psi, base_phi, s=2, c=colors)

    title = ax.set_title("Iteration: 0")
    plt.xlabel("φ")
    plt.ylabel("Ψ")

    # Update function for the animation
    def update(frame):
        nonlocal base_psi, base_phi, colors  # Ensure these variables are updated globally for each frame

        # Read new data for the current frame
        path = paths[frame]
        molecules, energies = read_xyz(path)
        new_psi = calculate_dihedrals_batch(molecules, indices1)
        new_phi = calculate_dihedrals_batch(molecules, indices2)

        # Append the new points to the cumulative lists
        base_psi = np.concatenate([base_psi, new_psi])
        base_phi = np.concatenate([base_phi, new_phi])

        # Update colors: set previous points to red, new points to blue
        colors = ["red"] * (len(base_psi) - len(new_psi)) + ["blue"] * len(new_psi)

        # Update scatter plot with new positions and colors
        scatter.set_offsets(np.c_[base_psi, base_phi])
        scatter.set_color(colors)

        title.set_text(f"Iteration: {frame + 1}")
        # Add legend
        red_patch = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markersize=5,
            label="Training",
            linestyle="None",
        )
        blue_patch = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markersize=5,
            label="Sampled",
            linestyle="None",
        )
        ax.legend(handles=[red_patch, blue_patch])

        return (scatter,)

    # Create the animation
    ani = FuncAnimation(
        fig, update, frames=len(paths), blit=False
    )  # Set blit=False for variable data

    # Save the animation as a GIF
    save_path = f"{"/".join(base_path.split("/")[:-3])}/plots/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    ani.save(save_path + "animation.gif", writer=PillowWriter(fps=fps))
    plt.close()


def get_al_animation(base_path):
    added_files = [f for f in os.listdir(base_path) if f.startswith("train")]
    num_files = len(added_files) - 1
    # paths.extend([base_path + f"train{i}.xyz" for i in range(num_files)])
    all_paths = os.listdir(base_path)
    paths = sorted(
        all_paths,
        key=lambda x: (
            int(x.split("train")[-1].split(".xyz")[0])
            if "train" in x and x.split("train")[-1].split(".xyz")[0].isdigit()
            else -1
        ),
    )
    paths = [os.path.join(base_path, p) for p in paths]

    paths = np.array(paths)

    # multi_scatter(paths[[-3,-2,-1]])
    # multi_scatter(paths[:])
    animate_active_learning(paths[0], paths[1:], fps=1)


def _plot_scatter(molecules, save_path):
    phi = calculate_dihedrals_batch(molecules, [6, 8, 14, 16])
    psi = calculate_dihedrals_batch(molecules, [4, 6, 8, 14])

    plt.figure(figsize=(8, 8))
    plt.scatter(psi, phi, s=2)
    plt.ylim(-180, 180)
    plt.xlim(-180, 180)
    plt.xlabel("φ")
    plt.ylabel("Ψ")
    plt.savefig(save_path)
    plt.close()


def get_space_coverage(base_path):
    added_files = [f for f in os.listdir(base_path) if f.startswith("train")]
    num_files = len(added_files) - 1
    # paths.extend([base_path + f"train{i}.xyz" for i in range(num_files)])
    all_paths = os.listdir(base_path)
    paths = sorted(
        all_paths,
        key=lambda x: (
            int(x.split("train")[-1].split(".xyz")[0])
            if "train" in x and x.split("train")[-1].split(".xyz")[0].isdigit()
            else -1
        ),
    )

    save_path = f"{"/".join(base_path.split("/")[:-3])}/plots/"
    paths = [os.path.join(base_path, p) for p in paths]
    molecules, _ = read_xyz(paths[0])
    print(np.array(molecules).shape)
    _plot_scatter(molecules, f"{save_path}/coverage_init.svg")
    for i, path in enumerate(paths[1:]):
        new_molecules, _ = read_xyz(path)
        molecules = np.concatenate([molecules, new_molecules], 0)
        _plot_scatter(molecules, f"{save_path}/coverage{i}.svg")


if __name__ == "__main__":
    # get_al_animation("al/run69/data/train/")
    get_space_coverage("al/run80/data/train/")
    get_space_coverage("al/run81/data/train/")
    get_space_coverage("al/run82/data/train/")
    get_space_coverage("al/run83/data/train/")
    get_space_coverage("al/run84/data/train/")
    get_space_coverage("al/run85/data/train/")
    get_space_coverage("al/run86/data/train/")
    get_space_coverage("al/run87/data/train/")
    get_space_coverage("al/run88/data/train/")
