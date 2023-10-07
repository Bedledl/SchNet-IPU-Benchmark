import os
from itertools import product
from os.path import join
from typing import Dict

import torch
from torch.utils.benchmark import Timer

from HalfPrecisionCalc import HalfPrecisionCalculator
from create_model import create_model

from ase.io.proteindatabank import read_proteindatabank
from ase.neighborlist import build_neighbor_list

from schnetpack.md import System
from schnetpack.ipu_modules.Calculator import BenchmarkCalculator

from shardedExecutionCalc import ShardedExecutionCalculator

# we use the configs of model_2 from
# https://github.com/torchmd/torchmd-net/blob/main/benchmarks/graph_network.ipynb

BENCHMARK_CONFIGS = {
    'embedding_dimension': 128,
    'num_layers': 6,
    'num_rbf': 50,
    'rbf_type': 'gauss',
    'trainable_rbf': False,
    'activation': 'ssp',
    'neighbor_embedding': False,
    'cutoff_lower': 0.0,
    'cutoff_upper': 5.0,
    'max_z': 100,
    'max_num_neighbors': 32,
    'aggr': 'add',
    'derivative': False,
    'atom_filter': -1,
    'prior_model': None,
    'output_model': 'Scalar',
    'reduce_op': 'add'
}


def torchmdnet_config_to_schnetpack_ipu_config(torchmdnet_config: Dict[str, str]):
    schnetpack_ipu_config = {
        "n_atom_basis": None,
        "n_rbf": None,
        "n_neighbors": None,
        "n_atoms": None,
        "n_batches": None,
        "rbf_cutoff": None,
        "n_interactions": None,
        "max_z": None,
    }

    equivalents = {
        'num_rbf': "n_rbf",
        'embedding_dimension': "n_atom_basis",
        'max_num_neighbors': "n_neighbors",
        "max_z": "max_z",
        'num_layers': "n_interactions",
        "cutoff_upper": "rbf_cutoff",
    }

    for key, val in torchmdnet_config.items():
        if key in equivalents.keys():
            schnetpack_ipu_config[equivalents[key]] = val
            continue

        if key == 'rbf_type':
            if  val != "gauss":
                raise ValueError("Currently only GaussianRBG is supported by this implementation. "
                                 "It is, however generally implemented in Schnetpack.")

        if key == 'trainable_rbf':
            if val != False:
                raise ValueError(
                    "Currently a trainable rbf is not supported by this implementation. "
                    "It is, however generally implemented in Schnetpack."
                )

        if key == "activation":
            if val != "ssp":
                raise ValueError(
                    "Currently only ShiftedSoftPlus is supported by this implementation. "
                    "It is, however generally implemented in Schnetpack."
                )

        # TODO
        if key == 'neighbor_embedding':
            pass#

        if key == 'cutoff_lower':
            pass

    return schnetpack_ipu_config


def benchmark(model, pdb_file):

    # Benchmark
    stmt = f'''
        energy = model()
        '''
    timer = Timer(stmt=stmt, globals=locals())
    speed = timer.blocked_autorange(min_run_time=10).median * 1000 # s --> ms
    it_s = 1000/speed

    return f"it/s: {it_s}, per run:{speed}"

def run_all_benchmarks(optimization: str = ""):
    PDB_FILES = os.getenv('PDB_FILES')
    if not PDB_FILES:
        raise ValueError("Please set the environment variable 'TORCHMD_NET' to the root directory"
                         " of a cloned version of the torchmd-net repository.")

    systems = [(join(PDB_FILES, 'alanine_dipeptide.pdb'), 'ALA2'),
               (join(PDB_FILES, 'chignolin.pdb'), "CLN"),
               (join(PDB_FILES, "villin.pdb"), "VIL"),
               (join(PDB_FILES, "profilin.pdb"), "PRO"),
               (join(PDB_FILES, "ferritin.pdb"), "FER"),
               (join(PDB_FILES, "dhfr.pdb"), "DHFR"),
               ]

    neighbors_regression_method = [torch.max]#[torch.mean, torch.max, torch.median]

    log_file = open("bechmark_result.log", "w")

    for system, calc_forces, regression_method in product(systems, [False, True], neighbors_regression_method):
        pdb_file, name = system

        mol = read_proteindatabank(pdb_file, index=0)

        system = System()
        system.load_molecules([mol])

        schnetpack_model_config = torchmdnet_config_to_schnetpack_ipu_config(BENCHMARK_CONFIGS)
        schnetpack_model_config["n_atoms"] = system.n_atoms
        schnetpack_model_config["n_batches"] = 1

        # we determine the k for the KNN neighborlist out of the numbers of neighors
        # that are computed with a cutoff radius
        nl = build_neighbor_list(mol, [schnetpack_model_config["rbf_cutoff"]]*len(mol))
        num_neighbors_cutoff = torch.tensor([len(nl.get_neighbors(i)[0]) - 1 for i in range(len(mol))],
                                            dtype=torch.float64)
        num_neighbors = regression_method(num_neighbors_cutoff)
        num_neighbors = min(int(num_neighbors), BENCHMARK_CONFIGS["max_num_neighbors"] )

        # nl contains self-loop -> decrement num_neighbors
        schnetpack_model_config["n_neighbors"] = num_neighbors
        print(f"max_neighbors: {num_neighbors}")

        schnetpack_model_config["calc_forces"] = calc_forces

        model = create_model(**schnetpack_model_config)

        if optimization == "":
            calculator_cls = BenchmarkCalculator
        elif optimization == "sharded":
            calculator_cls = ShardedExecutionCalculator
        elif optimization == "half":
            calculator_cls = HalfPrecisionCalculator
        elif optimization == "YES":
            raise NotImplementedError("Not implemented yet, but this should combine all optimizations.")


        calc = calculator_cls(
            model,
            "forces",  # force key
            "kcal/mol",  # energy units
            "Angstrom",  # length units
            energy_key="energy",  # name of potential energies
            required_properties=[],  # additional properties extracted from the model
            run_on_ipu=True,
            n_neighbors=schnetpack_model_config["n_neighbors"]
        )

        calc.compile_model(system)

        model_call = calc.get_model_call(system)

        try:
            speed = benchmark(model_call, pdb_file)
            description = f'  {name}: {speed} ms/it     with forces: {calc_forces}' \
                          f'      k: {num_neighbors}({str(regression_method.__name__)})'
            print(description)
            log_file.write(description)
        except Exception as e:
            print(e)
            description = f'  {name}: failed     with forces: {calc_forces}' \
                          f'      k: {num_neighbors}({str(regression_method.__name__)})'
            print(description)
            log_file.write(description)

    log_file.close()

if __name__ == '__main__':
    run_all_benchmarks()
