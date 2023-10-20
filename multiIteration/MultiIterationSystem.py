from typing import Union, List

import torch
from schnetpack.md import System


class MultiIterationSystem(System):
    def __int__(self):
        super(MultiIterationSystem, self).__init__()
        self.positions = None
        self.momenta = None
        self.forces = None
        self.stress = None

        self.n_molecules = 1

    def load_molecules(
        self,
        molecules: Union['Atoms', List['Atoms']],
        n_neighbors: int,
        n_replicas: int = 1,
        position_unit_input: Union[str, float] = "Angstrom",
        mass_unit_input: Union[str, float] = 1.0,
    ):
        super().load_molecules(molecules, n_replicas, position_unit_input, mass_unit_input)
        self.n_molecules = n_replicas
        self.n_neighbors = n_neighbors
        self.register_buffer("idx_i", torch.arange(self.total_n_atoms) \
            .repeat_interleave(self.n_neighbors))
        self.register_buffer("offsets", torch.tensor([[0, 0, 0]]) \
            .repeat(self.total_n_atoms * self.n_neighbors, 1))

        # this is copied from calculator. If we would want flxibitliy concerning n_replicas,
        # it is necessary to do this in the Calc. But we just want fast, fixed sized simulation.
        self.atom_types = self.atom_types.repeat(self.n_replicas)

        # Get n_atoms
        self.n_atoms = self.n_atoms.repeat(self.n_replicas)

        # Construct index vector for all replicas and molecules
        self.index_m = (
                self.index_m.repeat(self.n_replicas, 1)
                + self.n_molecules
                * torch.arange(self.n_replicas, device=self.device).long().unsqueeze(-1)
        ).view(-1)

        # Get cells and PBC
        self.cells = self.cells.view(-1, 3, 3)
        self.pbc = self.pbc.repeat(self.n_replicas, 1, 1).view(-1, 3)

    def get_initial_momenta(self, temperature):
        """
        Copied from build.lib.schnetpack.md.initial_conditions.UniformInit._setup_momenta
        :return: initial random momenta
        """
        momenta = torch.randn(self.n_replicas, self.total_n_atoms, 3) * self.masses

        # Scale velocities to desired temperature
        scaling = torch.sqrt(
            temperature[None, :, None]
            / temperature
        )
        momenta *= self.expand_atoms(scaling)
        return momenta