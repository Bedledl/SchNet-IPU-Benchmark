import torch
import poptorch

from functools import partial

from schnetpack.ipu_modules import BenchmarkCalculator
from schnetpack.md import System
from typing import Union, Dict, List

class ShardedExecutionCalculator(BenchmarkCalculator):
    """
    This calculator shards the SchNet Model
    """
    def __init__(
            self,
            model,
            force_key: str,
            energy_unit: Union[str, float],
            position_unit: Union[str, float],
            n_neighbors: int,
            energy_key: str = None,
            stress_key: str = None,
            required_properties: List = [],
            property_conversion: Dict[str, Union[str, float]] = {},
            run_on_ipu=True,
    ):
        super().__init__(
            model,
            force_key,
            energy_unit,
            position_unit,
            n_neighbors,
            energy_key,
            stress_key,
            required_properties,
            property_conversion
        )
        if not run_on_ipu:
            raise NotImplementedError("Sharded Execution is only available for IPU")

        model.eval()
        model.to(torch.float32)

        model.input_modules = torch.nn.ModuleList(
            [
                poptorch.BeginBlock(layer, ipu_id=0) for layer in model.input_modules
            ]
        )
        model.output_modules = torch.nn.ModuleList(
            [
                poptorch.BeginBlock(layer, ipu_id=1) for layer in model.output_modules
            ]
        )

        model.representation = poptorch.BeginBlock(model.representation, ipu_id=2)

        opts = poptorch.Options()
        # Automatically create 3 shards based on the block names
        opts.setExecutionStrategy(poptorch.ShardedExecution(poptorch.AutoStage.AutoIncrement))

        self.model = poptorch.inferenceModel(model, opts)

    def get_inputs(self, system):
        return self._get_system_molecules(system)

    def get_model_call(self, system):
        """This method returns a callable that runs the model and that can be used for a benchmark"""
        inputs = self._get_system_molecules(system)
        return partial(self.model, inputs)

    def calculate(self, system: System):
        inputs = self._get_system_molecules(system)
        return self.model(inputs)