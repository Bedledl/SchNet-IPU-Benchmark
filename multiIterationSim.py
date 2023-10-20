import os
from itertools import product
from os.path import join
from typing import Dict

import torch
from torch.utils.benchmark import Timer

import poptorch

from create_model import create_model, create_calculator

from ase.io.proteindatabank import read_proteindatabank
from ase.neighborlist import build_neighbor_list
from schnetpack.md import System as DummySystem, Simulator
from schnetpack.md.calculators import SchNetPackCalculator

from multiIteration.MultiIterationCalculator import MultiIterationCalculator
from multiIteration.MultiIterationSimulator import MultiIterationSimulator, MultiIterationVelocityVerlet
from multiIteration.MultiIterationSystem import MultiIterationSystem

iterations_per_call = 1000

schnetpack_ipu_config = {
        "n_atom_basis": 128,
        "n_rbf": 50,
        "n_batches": 1,
        "rbf_cutoff": 5.0,
        "n_interactions": 6,
        "max_z": 100,
}

temperature = torch.tensor([300])

def multi_iteration_simulation(pdb_file):

    mol = read_proteindatabank(pdb_file, index=0)

    system = MultiIterationSystem()
    system.load_molecules([mol], n_neighbors=schnetpack_ipu_config["n_neighbors"])

    schnetpack_ipu_config["n_atoms"] = system.n_atoms
    schnetpack_ipu_config["n_batches"] = 1

    model = create_model(**schnetpack_ipu_config)
    #we still want to have the inputs building method of the calculator
    calc = MultiIterationCalculator(model)

    integrator = MultiIterationVelocityVerlet(0.5)
    simulator = MultiIterationSimulator(system, integrator, calc)

    momenta = system.get_initial_momenta(temperature=temperature)
    positions = system.positions.view(-1, 3)
    # get initial forces
    energy, forces = calc.calculate(system, positions)
    system.positions = None

    model = poptorch.inferenceModel(simulator)
    return model, positions, momenta, forces


def benchmark(model, iterations_per_call, positions, momenta, forces):

    # Benchmark
    stmt = f'''
        energy = model(iterations_per_call, positions, momenta, forces)
        '''
    timer = Timer(stmt=stmt, globals=locals())
    speed = timer.blocked_autorange(min_run_time=10).median * 1000 # s --> ms
    it_s = 1000/speed

    return f"it/s: {it_s}, per run:{speed}"


def run_multi_iteration_benchmarks():
    PDB_FILES = os.getenv('PDB_FILES')
    if not PDB_FILES:
        print("The environment variable 'PDB_FILES' seems to be not set. Using default 'data/structures'.")
        PDB_FILES = 'data/structures'

    systems = [(join(PDB_FILES, 'alanine_dipeptide.pdb'), 'ALA2', 21),
               (join(PDB_FILES, 'chignolin.pdb'), "CLN", 32),
               (join(PDB_FILES, "villin.pdb"), "VIL", 32),
               (join(PDB_FILES, "profilin.pdb"), "PRO", 32),
               (join(PDB_FILES, "ferritin.pdb"), "FER", 32),
               (join(PDB_FILES, "dhfr.pdb"), "DHFR", 32),
               ]

    log_file = open("bechmark_result.log", "a")

    for molecule, calc_forces in product(systems, [True]):
        pdb_file, name, neighbors = molecule

        schnetpack_ipu_config["n_neighbors"] = neighbors
        schnetpack_ipu_config["calc_forces"] = calc_forces

        model_call, positions, momenta, forces = multi_iteration_simulation(pdb_file)

        model_call(iterations_per_call, positions, momenta, forces)

        try:
            speed = benchmark(model_call, iterations_per_call, positions, momenta, forces)
            description = f'  {name}: {speed} ms/it     with forces: {calc_forces}'
            print(description)
            log_file.write(description)
        except Exception as e:
            print(e)
            description = f'  {name}: failed     with forces: {calc_forces}'
            print(description)
            log_file.write(description)

    log_file.close()

if __name__ == '__main__':
    run_multi_iteration_benchmarks()


