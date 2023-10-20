from typing import Union, List, Dict

import torch
from schnetpack.md import System
from schnetpack.ipu_modules import BenchmarkCalculator, IPUCalculator
from schnetpack import properties

class MultiIterationCalculator(torch.nn.Module):
    def __init__(
            self,
            model
    ):
        super(MultiIterationCalculator, self).__init__()
        self.model = model

    def build_inputs(self, system: System):
        inputs = {
            properties.Z: system.atom_types,
            properties.n_atoms: system.n_atoms,
            properties.idx_m: system.index_m,
            properties.cell: system.cells,
            properties.pbc: system.pbc,
            properties.n_molecules: system.n_molecules,
            properties.idx_i: system.idx_i,
            properties.offsets: system.offsets,
        }

        return inputs

    def calculate(self, system, positions):
        inputs = self.build_inputs(system)
        inputs[properties.R] = positions
        outputs = self.model(inputs)
        return outputs["energy"], outputs["forces"]
