import functools
import os
import traceback
from itertools import product
from os.path import join
from typing import Dict
from torch import save, tensor, float32
from torch.utils.benchmark import Timer
import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.md.neighborlist_md import NeighborListMD
from schnetpack.transform import ASENeighborList

from torch.nn import Identity

from schnetpack.md.calculators import SchNetPackCalculator
from schnetpack import properties
from schnetpack.md import System

from ase.io.proteindatabank import read_proteindatabank

from GPUKNNNeighborlist import KNNNeighborTransform

schnetpack_config = {
        "n_atom_basis": 128,
        "n_rbf": 50,
        "n_neighbors": 21,
        "n_batches": 1,
        "rbf_cutoff": 5.0,
        "n_interactions": 6,
        "max_z": 100,
}


def create_gpu_model(
        n_atom_basis,
        n_rbf,
        n_neighbors,
        n_atoms,
        n_batches,
        max_z,
        n_interactions,
        rbf_cutoff,
        constant_batch_size=True,
        calc_forces=True,
        energy_key="energy",
        forces_key="forces",
):
    pred_energy = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=energy_key)
    if calc_forces:
        pred_forces = spk.atomistic.Forces(energy_key=energy_key, force_key=forces_key)
    else:
        pred_forces = Identity()

    pairwise_distance = spk.atomistic.PairwiseDistances()  # calculates pairwise distances between atoms
    radial_basis = spk.nn.GaussianRBF(n_rbf=n_rbf, cutoff=rbf_cutoff)
    schnet = spk.representation.SchNet(
        n_atom_basis=n_atom_basis, n_interactions=n_interactions,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(rbf_cutoff)
    )

    nnpot = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[pairwise_distance],
        output_modules=[pred_energy, pred_forces],
        postprocessors=[
        ]
    )

    return nnpot

def create_gpu_calculator(
        rbf_cutoff,
        cutoff_shell,
        model_path
):

    # KNNNeighborlistTransform takes different arguments then the other cutoff-based neighborlists
    # therefore we do a little magic here
    neighborlist_bound_args = functools.partial(
        KNNNeighborTransform,
        k=schnetpack_config["n_neighbors"],
        n_replicas=1,
        n_atoms=schnetpack_config["n_atoms"].item()
    )
    neighborlist_dummy = lambda cutoff: neighborlist_bound_args()

    md_neighborlist = NeighborListMD(
        rbf_cutoff,
        cutoff_shell,
        neighborlist_dummy,
    )
    md_neighborlist._filter_indices = lambda x, y: y

    md_calculator = SchNetPackCalculator(
        model_path,  # path to stored model
        "forces",  # force key
        "kcal/mol",  # energy units
        "Angstrom",  # length units
        md_neighborlist,  # neighbor list
        energy_key="energy",  # name of potential energies
        required_properties=[],  # additional properties extracted from the model
    )

    return md_calculator

def benchmark(model, system):

    # Benchmark
    stmt = f'''
        energy = model(system)
        '''
    timer = Timer(stmt=stmt, globals=locals())
    speed = timer.blocked_autorange(min_run_time=10).median * 1000 # s --> ms
    it_s = 1000/speed

    return f"it/s: {it_s}, per run:{speed}"

def run_all_benchmarks(optimization: str = ""):
    PDB_FILES = os.getenv('PDB_FILES')
    if not PDB_FILES:
        print("The environment variable 'PDB_FILES' seems to be not set. Using default 'data/structures'.")
        PDB_FILES = '../data/structures'

    systems = [(join(PDB_FILES, 'alanine_dipeptide.pdb'), 'ALA2'),
               (join(PDB_FILES, 'chignolin.pdb'), "CLN"),
               (join(PDB_FILES, "villin.pdb"), "VIL"),
               (join(PDB_FILES, "profilin.pdb"), "PRO"),
               (join(PDB_FILES, "ferritin.pdb"), "FER"),
               (join(PDB_FILES, "dhfr.pdb"), "DHFR"),
               ]

    log_file = open("bechmark_result.log", "w")

    for molecule, calc_forces in product(systems, [False, True]):
        pdb_file, name = molecule

        mol = read_proteindatabank(pdb_file, index=0)

        system = System()
        system.load_molecules([mol])

        schnetpack_config["n_atoms"] = system.n_atoms
        schnetpack_config["n_batches"] = 1

        schnetpack_config["calc_forces"] = calc_forces

        model = create_gpu_model(**schnetpack_config)
        save(model, "tmp_model")

        calc = create_gpu_calculator(schnetpack_config["rbf_cutoff"], 2.0, "tmp_model")
        calc.to("cuda")
        system.to("cuda")
        calc.to(float32)
        system.to(float32)

        def model_call(system):
            inputs = calc._generate_input(system)
            inputs["_offsets"] = tensor([[0, 0, 0]]) \
                .repeat(system.total_n_atoms * schnetpack_config["n_neighbors"], 1).to("cuda")

            return calc.model(inputs)

        try:
            speed = benchmark(model_call, system)
            description = f'  {name}: {speed} ms/it     with forces: {calc_forces}'
            print(description)
            log_file.write(description)
        except Exception as e:
            traceback.print_exc()
            description = f'  {name}: failed     with forces: {calc_forces}'
            print(description)
            log_file.write(description)

    log_file.close()

if __name__ == '__main__':
    run_all_benchmarks()

