import numpy as np
import torch
import os
import csv


class MD17Dataset(torch.utils.data.Dataset):
    def __init__(self, foldername, seed=42, subtract_self_energies=True, in_unit="kj/mol", train=None, train_ratio=0.8, scale=True, determine_norm=False, store_norm_path=None, load_norm_path=None):
        self.type_to_number = {"H" : 0,
                               "C" : 1,
                               "N" : 2,
                               "O" : 3}
        self.ground_charges  = {-1: 0.0,
                                0: 1.0,
                                1: 6.0,
                                2: 7.0,
                                3: 8.0}
        # energies in hartree
        self_energies = {-1: 0.0,
                         0: -0.500607632585,
                         1: -37.8302333826,
                         2: -54.5680045287,
                         3: -75.0362229210}
        self.Eh_to_eV = 27.211399
        self.kcal_to_eV = 0.0433641
        self.charge_scale = torch.tensor(max(self.ground_charges.values())+1)
        self.charge_power = 2
        if load_norm_path is not None:
            with open(load_norm_path, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if row[0] == "mean_energy":
                        self.mean_energy = float(row[1])
                    elif row[0] == "std_energy":
                        self.std_energy = float(row[1])
                    elif row[0] == "mean_forces":
                        self.mean_forces = float(row[1])
                    elif row[0] == "std_forces":
                        self.std_forces = float(row[1])
        else:
            self.mean_energy = -5.1623        # mean energy obtained from the training dataset (train_in)
            self.std_energy = 14.9431          # std energy obtained from the training dataset (train_in)
            self.mean_forces = self.mean_energy
            self.std_forces = self.std_energy
        
        self.atom_numbers_raw, self.coordinates_raw, self.energies, self.forces = self._read_coordinates_energies_forces(foldername, in_unit=in_unit)
        self.atom_numbers = self._pad_array(self.atom_numbers_raw, fill = [-1])
        self.coordinates = self._pad_array(self.coordinates_raw, fill = [[0, 0, 0]])

        self.atom_mask = torch.where(torch.tensor(self.atom_numbers) != -1, torch.tensor(1.0), torch.tensor(0.0))
    
        self.atom_numbers = np.array(self.atom_numbers)
        self.coordinates = np.array(self.coordinates)
        self.energies = np.array(self.energies)
        self.total_self_energy = 0
        if subtract_self_energies:
            self.total_self_energy = [sum(self_energies[atom] for atom in molecule) * self.Eh_to_eV for molecule in self.atom_numbers] 
            self.energies = self.energies - self.total_self_energy

        self.num_atoms = self.atom_numbers.shape[1]
        self.num_snapshots = self.atom_numbers.shape[0]
        self.num_types = len(self.type_to_number)

        self.one_hot = torch.eye(self.num_types, dtype=torch.float)[torch.tensor(self.atom_numbers.flatten())].reshape(-1, self.num_atoms, self.num_types)
        self.charges = torch.tensor([self.ground_charges[atom] for atom in self.atom_numbers.flatten()]).reshape(-1, self.num_atoms)

        #self.atom_mask = torch.ones((self.num_atoms), dtype=torch.float).reshape(-1, 1)
        self.edge_mask = torch.eye(self.num_atoms, dtype=torch.float).unsqueeze(0).expand(self.num_snapshots, -1, -1)
        self.edge_mask = torch.where(self.edge_mask == 1, torch.tensor(0.0), torch.tensor(1.0))
        self.edge_mask = self.edge_mask * self.atom_mask.unsqueeze(1) * self.atom_mask.unsqueeze(2)
        self.edge_mask = self.edge_mask.view(self.num_snapshots, -1)

        self.energies = torch.tensor(self.energies)
        self.forces = torch.tensor(self.forces)
        self.coordinates = torch.tensor(self.coordinates)

        # Shuffle the data
        np.random.seed(seed)
        indices = np.arange(self.num_snapshots)
        np.random.shuffle(indices)
        
        if train is not None:   
            num_train = int(train_ratio * self.num_snapshots)
            if train:  
                indices = indices[:num_train]
            else:
                indices = indices[num_train:]

        self.len = len(indices)

        

        self.atom_numbers = self.atom_numbers[indices]
        self.coordinates = self.coordinates[indices]
        self.energies = self.energies[indices]
        self.forces = self.forces[indices]
        self.one_hot = self.one_hot[indices]
        self.charges = self.charges[indices]
        self.atom_mask = self.atom_mask[indices]
        self.edge_mask = self.edge_mask[indices]

        if determine_norm:
            if load_norm_path is not None:
                print("WARNING: load_norm_path is not None, so the normalization parameters are not determined from dataset.")
            else:
                self.mean_energy = self.mean_forces = self.energies.mean()
                self.std_energy = self.std_forces = self.energies.std()
        if store_norm_path is not None:
            with open(store_norm_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["mean_energy", self.mean_energy.item()])
                writer.writerow(["std_energy", self.std_energy.item()])
                writer.writerow(["mean_forces", self.mean_forces.item()])
                writer.writerow(["std_forces", self.std_forces.item()])

        if scale:
            self.energies = (self.energies - self.mean_energy) / self.std_energy
            self.forces = (self.forces - self.mean_forces) / self.std_forces
            

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return {"one_hot" : self.one_hot[idx],
                "coordinates" : self.coordinates[idx],
                "energies" : self.energies[idx],
                "atom_mask": self.atom_mask[idx],
                "edge_mask": self.edge_mask[idx],
                "charges": self.charges[idx],
                "forces": self.forces[idx],
                "charge_scale": self.charge_scale,
                "charge_power": self.charge_power,
                }
    

    def _read_coordinates_energies_forces(self, foldername, in_unit="kcal/mol"):
        all_atom_numbers = []
        all_coordinates = []
        all_forces = []
        energies = []
        for filename in os.listdir(foldername):
            if filename.endswith(".xyz"):
                with open(os.path.join(foldername, filename), 'r') as file:
                    while True:
                        line = file.readline()
                        if line == "":
                            break
                        num_atoms = int(line.strip())
                        


                        # Skip the comment line
                        if in_unit == "kcal/mol":
                            energy = float(file.readline().strip()) * self.kcal_to_eV
                        elif in_unit == "kj/mol":
                            energy = float(file.readline().strip())
                        energies.append(energy)

                        # Initialize lists to store atom types and coordinates
                        atom_numbers = []

                        coordinates = []
                        forces = []

                        # Read atom types and coordinates for each atom
                        for _ in range(num_atoms):
                            line = file.readline().split()
                            atom_type = line[0]
                            x, y, z = map(float, line[1:4])
                            fx, fy, fz = map(float, line[4:7])

                            atom_numbers.append(int(self.type_to_number[atom_type]))
                            coordinates.append([x, y, z]) 
                            forces.append([fx, fy, fz])
                        all_coordinates.append(coordinates)
                        all_atom_numbers.append(atom_numbers)
                        all_forces.append(forces)

        return all_atom_numbers, all_coordinates, energies, all_forces
    

    def _pad_array(self, arr, fill = [-1]):
        max_length = max(len(nums) for nums in arr)
        padded_arr = []
        for nums in arr:
            # Calculate the number of padding elements needed
            num_padding = max_length - len(nums)
            # Pad the atom numbers with the filling element
            padded_nums = nums + fill * num_padding
            padded_arr.append(padded_nums)

    
        return padded_arr






if __name__ == "__main__":
    ds = MD17Dataset("datasets/files/ala_converged_forces_1000", subtract_self_energies=False, train=True)
    
