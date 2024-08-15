import torch
import numpy as np
import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as unit

# Define a simple PyTorch model for energy prediction
class SimpleEnergyModel(torch.nn.Module):
    def __init__(self):
        super(SimpleEnergyModel, self).__init__()
        # Define a simple linear layer (for example purposes)
        
    def forward(self, positions):
        # Compute pairwise distances between particles
        distances = torch.cdist(positions, positions, p=2)
        # Square the distances
        squared_distances = torch.square(distances)
        # Sum the squared distances
        energy = torch.sum(squared_distances)
        return energy

# Instantiate the model
model = SimpleEnergyModel()

def compute_forces_from_energy(positions):
    # Convert OpenMM positions to PyTorch tensor
    positions_tensor = torch.tensor(positions, requires_grad=True, dtype=torch.float32)
    
    # Forward pass through the model to compute energy
    energy = model(positions_tensor).sum()  # Sum over the energies if the output is a vector
    print(energy)
    # Compute gradients (forces)
    energy.backward()
    forces = -positions_tensor.grad.detach().numpy()  # Convert back to numpy array
    
    return forces

# Create the system and add particles
system = mm.System()
num_particles = 10  # Example with 10 particles
for i in range(num_particles):
    system.addParticle(1.0 * unit.dalton)  # Adding particles with mass 1 dalton

# Define a custom force with a dummy expression
custom_force = mm.CustomExternalForce('0.0')  # The expression here is just a placeholder
for i in range(num_particles):
    custom_force.addParticle(i, [])
system.addForce(custom_force)



# Set up an integrator and context
temperature = 300 * unit.kelvin
friction = 1 / unit.picosecond
timestep = 0.001 * unit.picoseconds
integrator = mm.LangevinIntegrator(temperature, friction, timestep)
platform = mm.Platform.getPlatformByName('CPU')


# Set initial positions (randomly in this case)


# Create a topology element
topology = app.Topology()
chain = topology.addChain()
for i in range(num_particles):
    residue = topology.addResidue("RES", chain)
    atom = topology.addAtom("ATOM", app.Element.getBySymbol('C'), residue)
    topology.addBond(atom, atom)  # Add a bond between the same atom to satisfy topology requirements

simulation = app.Simulation(topology, system, integrator, platform)
context = simulation.context
initial_positions = np.random.randn(num_particles, 3) * 0.1  # Random positions centered around 0
context.setPositions(initial_positions * unit.nanometers)

# Minimize the energy before running the simulation
mm.LocalEnergyMinimizer.minimize(context)

# Run the simulation and apply the custom forces at each step
n_steps = 10000
for step in range(n_steps):
    state = simulation.context.getState(getPositions=True, getForces=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
    
    
    # Calculate custom forces using the PyTorch model
    forces = compute_forces_from_energy(positions)
    

    
    simulation.context.setState(state)
    # Update custom force parameters
    for i in range(num_particles):
        custom_force.setParticleParameters(i, i, [forces[i, 0], forces[i, 1], forces[i, 2]])

    custom_force.updateParametersInContext(context)
    state = simulation.context.getState(getPositions=True, getForces=True)
    mm_forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometers)
    old_pos = positions
    # Integrate the system
    simulation.step(1)

    state = context.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
    pos_dif = positions - old_pos
    
    # Optionally print energy, positions, etc. for each step
    #if step % 100 == 0:
    #    print(model(torch.Tensor(positions)))

# Final state output
state = context.getState(getPositions=True, getEnergy=True)
print(f"Final Potential Energy = {state.getPotentialEnergy()}")

