from typing import List

import torch
import poptorch


def log(logs_str):
    print(logs_str)


class WrapperGradient(torch.nn.Module):
    """wraps a model and calculates the dervative w.r.t the n-th input."""
    def __init__(self, model, n=0):
        super().__init__()
        self.inner_model = model
        self.n = n

    def forward(self, *args):
        args[self.n].requires_grad_()
        result_model = self.inner_model(*args)
        result_grad = torch.autograd.grad(
            result_model,
            args[self.n],
            torch.ones_like(result_model)
        )[0]
        return result_model, result_grad


class Runner:
    def __init__(self, op_name: str):
        self.op_name = op_name
    def run_inner(self):
        raise NotImplementedError

    def run(self):
        log("-"*80)
        log("")
        self.run_inner()
        log("")
        log("")



class BackwardNotSupported(Runner):
    """
    Compiles the operation for IPU and verifies
    that its backward pass requires unsupported operations.
    """
    def __init__(self, op_name: str, operation_model: torch.nn.Module, input: List, n: int):
        super().__init__(op_name)
        self.op_model = operation_model
        self.input = input
        self.n = n

    def run_inner(self):
        log(f"Verify that {self.op_name} calls unsupported operations, if the gradient w.r.t. an input is computed:")

        # wrap the model in the gradient model and poplar executor
        model = WrapperGradient(self.op_model, self.n)
        ipu_model = poptorch.inferenceModel(model)

        model(*self.input)
        log(f"  Model with gradient computation run without errors on the CPU.")

        try:
            ipu_model(*self.input)
        except Exception as exc:
            log(f"  Exception occurred during run on IPU: {exc}")


class ReplacementIsEquivalent(Runner):
    """
    runs the replaced operation(A) and the replacement(B) operation on the IPU
    and compares the result
    """
    def __init__(self, op_name: str,
                 model_A: torch.nn.Module, input_A: List,
                 model_B: torch.nn.Module, input_B: List):
        super().__init__(op_name)
        self.model_A = model_A
        self.model_B = model_B
        self.input_A = input_A
        self.input_B = input_B

    def run_inner(self):
        log(f"Verify that the replaced operation and the replacement ({self.op_name}) have the same result")

        model_A_ipu = poptorch.inferenceModel(self.model_A)
        model_B_ipu = poptorch.inferenceModel(self.model_B)

        assert torch.all(model_A_ipu(*self.input_A) == model_B_ipu(*self.input_B))


class GradientIsEquivalent(Runner):
    """
    runs the replaced operation + gradient computation on the CPU(model A)
    and the replacement operation on the IPU(model B. Then compares the results.
    """
    def __init__(self, op_name: str,
                 model_A: torch.nn.Module, input_A: List, n_A: int,
                 model_B: torch.nn.Module, input_B: List, n_B: int):
        super().__init__(op_name)
        self.model_A = model_A
        self.model_B = model_B
        self.input_A = input_A
        self.input_B = input_B
        self.n_A = n_A
        self.n_B = n_B

    def run_inner(self):
        log(f"Verify that the gradient of the replaced operation and the replacement ({self.op_name}) are the same.")

        grad_model_A = WrapperGradient(self.model_A, self.n_A)
        grad_model_B = WrapperGradient(self.model_B, self.n_B)

        grad_model_B = poptorch.inferenceModel(grad_model_B)

        _, grad_A = grad_model_A(*self.input_A)
        _, grad_B = grad_model_B(*self.input_B)

        assert torch.all(grad_A == grad_B)



class GradientIsNull(Runner):
    """
    Runs the operations and gradient computation on the IPU and verifies that is output is always zero.
    """
    pass
