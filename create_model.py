import schnetpack as spk
import schnetpack.transform as trn
import schnetpack.ipu_modules as schnet_ipu_modules
from schnetpack.ipu_modules.Calculator import BenchmarkCalculator

from torch.nn import Identity

from AllOptimizationCalc import AllOptimizationCalculator
from HalfPrecisionCalc import HalfPrecisionCalculator
from shardedExecutionCalc import ShardedExecutionCalculator


def create_model(
        n_atom_basis,
        n_rbf,
        n_neighbors,
        n_atoms,
        n_batches,
        max_z,
        rbf_cutoff=5.,
        n_interactions=3,
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

    neighbor_distance = schnet_ipu_modules.KNNNeighborTransform(n_neighbors, n_batches, n_atoms,
                                                                always_update=True)

    pairwise_distance = schnet_ipu_modules.PairwiseDistancesIPU()

    radial_basis = schnet_ipu_modules.GaussianRBFIPU(n_rbf=n_rbf, cutoff=rbf_cutoff)

    schnet = spk.representation.SchNet(
        n_atom_basis=n_atom_basis, n_interactions=n_interactions,
        radial_basis=radial_basis,
        max_z=max_z,
        cutoff_fn=schnet_ipu_modules.CosineCutoff(rbf_cutoff),
        activation=schnet_ipu_modules.ShiftedSoftplusIPU(),
        n_neighbors=n_neighbors,
    )

    nnpot = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[neighbor_distance, pairwise_distance],
        output_modules=[pred_energy, pred_forces],
        postprocessors=[
        ]
    )

    return nnpot


def create_calculator(model, k, optimization):
    if optimization == "":
        calculator_cls = BenchmarkCalculator
    elif optimization == "sharded":
        calculator_cls = ShardedExecutionCalculator
    elif optimization == "half":
        calculator_cls = HalfPrecisionCalculator
    elif optimization == "YES":
        calculator_cls = AllOptimizationCalculator

    calc = calculator_cls(
        model,
        "forces",  # force key
        "kcal/mol",  # energy units
        "Angstrom",  # length units
        energy_key="energy",  # name of potential energies
        required_properties=[],  # additional properties extracted from the model
        run_on_ipu=True,
        n_neighbors=k
    )

    return calc
