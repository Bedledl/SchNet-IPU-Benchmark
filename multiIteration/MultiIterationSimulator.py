import poptorch
import torch
import torch.nn as nn
from contextlib import nullcontext

from tqdm import trange

from schnetpack.md import System


class MultiIterationVelocityVerlet(nn.Module):
    def __init__(self, time_step):
        super(MultiIterationVelocityVerlet, self).__init__()
        self.time_step = time_step

    def main_step(self, system: System, old_positions, momenta):
        """
        Propagate the positions of the system according to:

        ..math::
            q = q + \frac{p}{m} \delta t
        :return: new positions
        """
        new_positions = old_positions + self.time_step * momenta / system.masses
        return new_positions.view(-1, 3)

    def half_step(self, momenta, forces):
        """
        Half steps propagating the system momenta according to:

        ..math::
            p = p + \frac{1}{2} F \delta t

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        :returns new momenta
        """
        return momenta + 0.5 * forces * self.time_step


class MultiIterationSimulator(nn.Module):
    """

    """

    def __init__(
        self,
        system: System,
        integrator,
        calculator,
        simulator_hooks: list = [],
        step: int = 0,
        restart: bool = False,
        gradients_required: bool = False,
        progress: bool = True,
    ):
        super(MultiIterationSimulator, self).__init__()

        self.system = system
        self.integrator = integrator
        self.calculator = calculator
        self.simulator_hooks = torch.nn.ModuleList(simulator_hooks)
        self.gradients_required = gradients_required


    @property
    def device(self):
        return self.system.device

    @property
    def dtype(self):
        return self.system.dtype

    def simulation_body(self, positions, momenta, forces):
        # Call hook before first half step
        for hook in self.simulator_hooks:
            hook.on_step_begin(self)

        # Do half step momenta
        momenta = self.integrator.half_step(momenta, forces)

        # Do propagation MD/PIMD
        positions = self.integrator.main_step(self.system, positions, momenta)

        # Compute new forces
        energy, forces = self.calculator.calculate(self.system, positions)

        # Call hook after forces
        for hook in self.simulator_hooks:
            hook.on_step_middle(self)

        # Do half step momenta
        momenta = self.integrator.half_step(momenta, forces)

        # Call hooks after second half step
        # Hooks are called in reverse order to guarantee symmetry of
        # the propagator when using thermostat and barostats
        for hook in self.simulator_hooks[::-1]:
            hook.on_step_end(self)

        # Logging hooks etc
        for hook in self.simulator_hooks:
            hook.on_step_finalize(self)

        return positions, momenta, forces

    def forward(self, n_steps: int, positions, momenta, forces):
        """
        Main simulation function. Propagates the system for a certain number
        of steps.

        Args:
            n_steps (int): Number of simulation steps to be performed.
        """
        self.system.to(self.device)

        # Check, if computational graph should be built
        if self.gradients_required:
            grad_context = torch.no_grad()
        else:
            grad_context = nullcontext()

        with grad_context:
            # Perform initial computation of forces

            # Call hooks at the simulation start
            for hook in self.simulator_hooks:
                hook.on_simulation_start(self)

            positions, _, _ = poptorch.for_loop(n_steps, self.simulation_body, [positions, momenta, forces])


            # Call hooks at the simulation end
            for hook in self.simulator_hooks:
                hook.on_simulation_end(self)

        return positions, momenta, forces
