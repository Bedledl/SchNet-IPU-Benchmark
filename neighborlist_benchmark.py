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

# we use the configs of model_2 from
# https://github.com/torchmd/torchmd-net/blob/main/benchmarks/graph_network.ipynb


schnetpack_ipu_config = {
        "n_atom_basis": 128,
        "n_rbf": 50,
        "n_neighbors": 3,
        "n_atoms": None,
        "n_batches": 1,
        "rbf_cutoff": 5.0,
        "n_interactions": 3,
        "max_z": 100,
}
max_num_neighbors = 32

def benchmark(model, pdb_file):

    # Benchmark
    stmt = f'''
        energy = model()
        '''
    timer = Timer(stmt=stmt, globals=locals())
    speed = timer.blocked_autorange(min_run_time=10).median * 1000 # s --> ms

    return speed

def run_all_benchmarks():
    torchmdnet_path = join(os.getenv('TORCHMD_NET'), "benchmarks/systems")
    if not torchmdnet_path:
        raise ValueError("Please set the environment variable 'TORCHMD_NET' to the root directory"
                         " of a cloned version of the torchmd-net repository.")

    systems = [(join(torchmdnet_path, 'alanine_dipeptide.pdb'), 'ALA2'),
               (join(torchmdnet_path, 'chignolin.pdb'), 'CLN'),
               (join(torchmdnet_path, 'dhfr.pdb'), 'DHFR'),
               (join(torchmdnet_path, 'factorIX.pdb'), 'FC9'),
               (join(torchmdnet_path, 'stmv.pdb'), 'STMV')]

    neighbors_method = torch.max

    log_file = open("bechmark_result.log", "w")

    for system, knn_on_ipu in product(systems, [True, False]):
        knn_module = None
        pdb_file, name = system

        mol = read_proteindatabank(pdb_file, index=0)

        system = System()
        system.load_molecules([mol])

        schnetpack_ipu_config["n_atoms"] = system.n_atoms
        schnetpack_ipu_config["n_batches"] = 1

        # we determine the k for the KNN neighborlist out of the numbers of neighors
        # that are computed with a cutoff radius
        nl = build_neighbor_list(mol, [schnetpack_ipu_config["rbf_cutoff"]]*len(mol))
        num_neighbors_cutoff = torch.tensor([len(nl.get_neighbors(i)[0]) - 1 for i in range(len(mol))],
                                            dtype=torch.float64)
        num_neighbors = neighbors_method(num_neighbors_cutoff)
        num_neighbors = min(int(num_neighbors), max_num_neighbors )

        # nl contains self-loop -> decrement num_neighbors
        schnetpack_ipu_config["n_neighbors"] = num_neighbors
        print(f"n_neighbors: {num_neighbors}")

        schnetpack_ipu_config["calc_forces"] = False

        model = create_model(**schnetpack_ipu_config)


        if not knn_on_ipu:
            #remove KNN module
            new_input_modules = []
            for input_module in model.input_modules:
                if input_module.startswith("KNNNeighborTransform"):
                    print("removed KNNNeighborTransform from input modules")
                    knn_module = input_module
                    continue
                new_input_modules.append(input_module)
            model.input_modules = torch.nn.ModuleList(new_input_modules)


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

        inputs = calc.get_inputs(system)

        if knn_on_ipu:
            model_call = calc.model
        else:
            model_call = lambda in_: calc.model(knn_module(in_))


        try:
            speed = benchmark(model_call, pdb_file, inputs)
            description = f'  {name}: {speed} ms/it     KNN on IPU: {knn_on_ipu}' \
                          f'      k: {num_neighbors}'
            print(description)
            log_file.write(description)
        except Exception as e:
            print(e)
            description = f'  {name}: failed     KNN on IPU: {knn_on_ipu}' \
                          f'      k: {num_neighbors}'
            print(description)
            log_file.write(description)

        break

    log_file.close()

if __name__ == '__main__':
    run_all_benchmarks()
