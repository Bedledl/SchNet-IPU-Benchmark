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

        opts = poptorch.Options()
        # Automatically create 3 shards based on the block names
        opts.setExecutionStrategy(poptorch.ShardedExecution(poptorch.AutoStage.AutoIncrement))

        output_module = []
        for layer in model.output_modules:
            if str(layer).startswith("Forces"):
                layer = poptorch.BeginBlock(layer)

            output_module.append(layer)

        model.output_modules = torch.nn.ModuleList(output_module)

#        interactions = []
#        for interaction in model.representation.interactions:
#            interaction = poptorch.BeginBlock(interaction)
#            interactions.append(interaction)

#        model.representation.interactions = torch.nn.ModuleList(interactions)

        self.model = poptorch.inferenceModel(model, opts)
