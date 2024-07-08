import os
from ase import Atoms
from ase.io import read
from ase.calculators.turbomole import Turbomole
import pandas as pd
import argparse
import concurrent.futures


def AddStatementToControl(controlfilename, statement):
    inf = open(controlfilename, 'r')
    lines = inf.readlines()
    inf.close()
    already_in=False
    outf = open(controlfilename, 'w')
    for line in lines:
        if statement.split()[0] in line:
            already_in=True
        if len(line.split()) > 0:
            if line.split()[0] == "$end" and not already_in:
                outf.write("%s\n" % (statement))
        outf.write(line)
    outf.close()


def turbomole(xyz_path, output_dir='turbomole_calc', functional='b3-lyp', basis_set='def2-SVP'):
    # Read the atomic structure from the XYZ file
    atoms = read(xyz_path)

    # Create a directory for the calculation
    calc_dir = output_dir
    if not os.path.exists(calc_dir):
        os.makedirs(calc_dir)
    os.chdir(calc_dir)

    params = {
        'density functional': functional,
        'basis set name': basis_set,
        'multiplicity': 1
    }

    calculator = Turbomole(**params)

    # Attach the calculator to the atoms
    atoms.calc = calculator

    #initialize the calculator to obtain the control file
    calculator.initialize()

    scratch_statement = f"$scratch files\n    dscf  dens  {calc_dir}/dens\n    dscf  fock  {calc_dir}/fock\n    dscf  dfock  {calc_dir}/dfock\n    dscf  ddens  {calc_dir}/ddens\n    dscf  statistics  {calc_dir}/statistics\n    dscf  errvec  {calc_dir}/errvec\n    dscf  oldfock  {calc_dir}/oldfock\n    dscf  oneint  {calc_dir}/oneint"

    AddStatementToControl(os.path.abspath('control'), scratch_statement)
    # Run the DFT calculation to get the orbital energies
    calculator.calculate(atoms)

    ## add scratch_file manually here
    os.system(f'eiger >> eiger.out')

    with open('eiger.out') as f:
        lines = f.readlines()
        for line in lines:
            if 'HOMO:' in line:
                homo = float(line.split()[-2])
            elif 'LUMO:' in line:
                lumo = float(line.split()[-2])

    if calculator.converged:
        print(f'The calculation converged!')
    else:
        print(f'The calculation did not converge!')

    print(f'Results: HOMO = {homo} eV, LUMO = {lumo} eV')

    os.chdir('..')

    return homo, lumo

def process_file(file, input_directory, node_directory):
    product_code = file[:-4]
    try:
        homo, lumo = turbomole(input_directory + '/' + file, node_directory + '/Results/'+ product_code + '_results')

    except Exception as e:
        print(f"Error processing {file}: {e}")
        homo, lumo = None, None
    return product_code, homo, lumo

def process_directory(input_directory, node_directory):
    print(f"Processing directory: {input_directory}")
    files = [file for file in os.listdir(input_directory) if file.endswith('.xyz')]
    product_codes, homos, lumos = [], [], []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_file, file, input_directory, node_directory): file for file in files}
        for future in concurrent.futures.as_completed(futures):
            product_code, homo, lumo = future.result()
            product_codes.append(product_code)
            homos.append(homo)
            lumos.append(lumo)

    return product_codes, homos, lumos

if __name__ == "__main__":
    """ parser = argparse.ArgumentParser(description="Process a directory.")
    parser.add_argument("directory", type=str, help="The directory to process")
    parser.add_argument("temp_node_dir", type=str, help="The assigned node temporary directory path")
    args = parser.parse_args()
    """
    filepath = os.path.dirname(os.path.abspath(__file__))
    in_dir = filepath + "/xyz_in/"
    tmp_dir = filepath + "/tmp"
    product_codes, homos, lumos = process_directory(in_dir, tmp_dir)

    df = pd.DataFrame()
    print(df.keys)
    """  df['Product Code'] = product_codes
    df['HOMO(eV)'] = homos
    df['LUMO(eV)'] = lumos
    df.to_csv(args.temp_node_dir + '/Results/DFT_results.csv') """
