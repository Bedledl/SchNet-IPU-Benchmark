import os
from os.path import join

from ase.io.proteindatabank import read_proteindatabank

from schnetpack.md import System

from create_model import create_calculator, create_model

PDB_FILES = os.getenv('PDB_FILES')
if not PDB_FILES:
    print("The environment variable 'PDB_FILES' seems to be not set. Using default 'data/structures'.")
    PDB_FILES = 'data/structures'

pdb_file = join(PDB_FILES, 'alanine_dipeptide.pdb')

mol = read_proteindatabank(pdb_file, index=0)

system = System()
system.load_molecules([mol])


schnetpack_ipu_config = {
        "n_atom_basis": 128,
        "n_rbf": 50,
        "n_neighbors": 21,
        "n_atoms": system.n_atoms,
        "n_batches": 1,
        "rbf_cutoff": 5.0,
        "n_interactions": 6,
        "max_z": 100,
}


def profiling(calculate_forces, report_dir: str, optimization):

    os.environ["POPLAR_ENGINE_OPTIONS"] = '{"autoReport.all":"true", "autoReport.directory":"'+report_dir+'"}'

    schnetpack_ipu_config["calc_forces"] = calculate_forces

    model = create_model(**schnetpack_ipu_config)
    calc = create_calculator(model, schnetpack_ipu_config["n_neighbors"], optimization)

    calc.compile_model(system)
    model_call = calc.get_model_call(system)
    result = model_call()
    print(result)


def profiling_original_schnetpack(report_dir: str):
    from schnetpack.md.calculators import SchNetPackCalculator
    from schnetpack.atomistic import Atomwise
    from schnetpack.representation import SchNet
    from schnetpack.ipu_modules import KNNNeighborTransform, PairwiseDistancesIPU, GaussianRBFIPU, DummyCutoff,\
        ShiftedSoftplusIPU, Cutoff

    os.environ["POPLAR_ENGINE_OPTIONS"] = '{"autoReport.all":"true", "autoReport.directory":"' + report_dir + '"}'

    pred_energy = Atomwise(n_in=n_atom_basis, output_key=energy_key)

    neighbor_distance = KNNNeighborTransform(n_neighbors, n_batches, n_atoms)

    pairwise_distance = PairwiseDistancesIPU()

    radial_basis = GaussianRBFIPU(n_rbf=n_rbf, cutoff=rbf_cutoff)

    schnet = SchNet(
         n_atom_basis=schnetpack_ipu_config["n_atom_basis"],
         n_interactions=schnetpack_ipu_config["n_interactions"],
         radial_basis=radial_basis,
         max_z=schnetpack_ipu_config["max_z"],
         cutoff_fn=Cutoff(schnetpack_ipu_config["rbf_cutoff"]),
         activation=ShiftedSoftplusIPU(),
         n_neighbors=schnetpack_ipu_config["n_neighbors"],
    )
#
#    nnpot = spk.model.NeuralNetworkPotential(
#        representation=schnet,
#        input_modules=[trn.CastTo32(), neighbor_distance, pairwise_distance],
#        output_modules=[pred_energy, pred_forces],
#        postprocessors=[
#            trn.CastTo64(),
#        ]
#    )
#
#    calc = SchNetPackCalculator(model_path,  # path to stored model
#                "forces",  # force key
#                "kcal/mol",  # energy units
#                "Angstrom",  # length units
#                md_neighborlist,  # neighbor list
#                energy_key="energy",  # name of potential energies
#                required_properties=[]  # additional properties extracted from the model)
#
#

profiling(True, "./profiling/with_checkpoints_ala")
#profiling(False, "./profiling/without_forces")
#profiling_sharded_model(True, "./profiling/sharded_without_checkpoints7_10")
