import os
from itertools import product
from os.path import join
from typing import Dict


import torch
from torch.utils.benchmark import Timer

from create_model import create_model

from ase.io.proteindatabank import read_proteindatabank
from ase.neighborlist import build_neighbor_list

from schnetpack.md import System
from schnetpack.ipu_modules.Calculator import BenchmarkCalculator

torchmdnet_path = join(os.getenv('TORCHMD_NET'), "benchmarks/systems")
if not torchmdnet_path:
    raise ValueError("Please set the environment variable 'TORCHMD_NET' to the root directory"
                     " of a cloned version of the torchmd-net repository.")

pdb_file = join(torchmdnet_path, 'chignolin.pdb')

mol = read_proteindatabank(pdb_file, index=0)

system = System()
system.load_molecules([mol])


schnetpack_ipu_config = {
        "n_atom_basis": 128,
        "n_rbf": 50,
        "n_neighbors": 15,
        "n_atoms": system.n_atoms,
        "n_batches": 1,
        "rbf_cutoff": 5.0,
        "n_interactions": 6,
        "max_z": 100,
    }


def profiling(calculate_forces, report_dir: str):
    os.environ["POPLAR_ENGINE_OPTIONS"] = '{"autoReport.all":"true", "autoReport.directory":"'+report_dir+'"}'

    schnetpack_ipu_config["calc_forces"] = calculate_forces

    model = create_model(**schnetpack_ipu_config)

    calc = BenchmarkCalculator(
        model,
        "forces",  # force key
        "kcal/mol",  # energy units
        "Angstrom",  # length units
        energy_key="energy",  # name of potential energies
        required_properties=[],  # additional properties extracted from the model
        run_on_ipu=True,
        n_neighbors=schnetpack_ipu_config["n_neighbors"]
    )

    calc.compile_model(system)

    model_call = calc.get_model_call(system)

    model_call()

profiling(True, "./profiling/with_forces")
profiling(False, "./profiling/without_forces")
