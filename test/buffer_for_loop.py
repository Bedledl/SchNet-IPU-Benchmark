import torch
import poptorch

class BufferPopTorchLoopModel(torch.nn.Module):
    def __init__(self):
        super(BufferPopTorchLoopModel, self).__init__()
        self.register_buffer("buf", torch.zeros((10, 3), dtype=torch.float32))

    def update(self, update_tensor):
        update_tensor = poptorch.nop(update_tensor)
        self.buf.copy_(update_tensor + self.buf)
        return update_tensor

    def forward(self, count, update_tensor):
        poptorch.for_loop(count, self.update, [update_tensor])
        return self.buf

class BufferPythonLoopModel(torch.nn.Module):
    def __init__(self):
        super(BufferPythonLoopModel, self).__init__()
        self.register_buffer("buf", torch.zeros((10, 3), dtype=torch.float32))

    def update(self, update_tensor):
        update_tensor = poptorch.nop(update_tensor)
        self.buf.copy_(update_tensor + self.buf)
        return update_tensor

    def forward(self, count, update_tensor):
        for _ in range(count):
            update_tensor = self.update(update_tensor)
        return self.buf

model_py = BufferPythonLoopModel()
model_pop = BufferPopTorchLoopModel()
model_py = poptorch.inferenceModel(model_py)
model_pop = poptorch.inferenceModel(model_pop)

print(model_py(4, torch.ones((10, 3), dtype=torch.float32)))

try:
    print(model_pop(4, torch.ones((10, 3), dtype=torch.float32)))
except Exception as e:
    print(f"As expected, an exception occurred when altering a buffer within the body of a poptorch.for_loop: {e}")

