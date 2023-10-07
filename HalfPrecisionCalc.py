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
        model.half()

        opts = poptorch.Options()
        # Automatically create 3 shards based on the block names
        opts.Precision.enableStochasticRounding(True)

        self.model = poptorch.inferenceModel(model, opts)

    def _get_system_molecules(self, system: System):
        inputs = super()._get_system_molecules(system)
        for key, val in inputs.items():
            if not isinstance(val, torch.Tensor):
                continue
            inputs[key] = val.half()
        return inputs