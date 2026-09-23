"""Log per-module and global RMS at AFTER_BACKWARD on steps 0, 100, ... .

Global activation RMS pools leaf-module outputs, weighted by element count.
Parameter-free leaves (e.g. rotary embeddings, whose outputs are constant
position tables) are skipped. Global parameter/gradient RMS pools unique model
parameters. Module parameter metrics include directly owned parameters only;
gradients are measured pre-clip.

Enable with MetricLogger(AFTER_BACKWARD, log_module_rms) in
TrainConfig.metric_loggers (requires wandb_online=True).
"""

import torch


class ModuleRMSLogger:
    def __init__(self, every=100):
        self.every = every
        self.handles = []
        self.samples = {}
        self.capture = False

    def setup(self, ctx):
        self.close()
        self.capture = ctx.step % self.every == 0
        for name, module in ctx.model.named_modules():
            is_leaf = not any(module.children())
            if is_leaf and next(module.parameters(recurse=False), None) is None:
                continue  # e.g. rotary embeddings

            def hook(module, inputs, output, name=name or "root", is_leaf=is_leaf):
                if self.capture and module.training:
                    outputs = output if isinstance(output, (tuple, list)) else (output,)
                    samples = self.record(f"rms/{name}/activation", outputs)
                    if is_leaf:
                        self.samples.setdefault("rms/global/activation", []).extend(samples)

            self.handles.append(module.register_forward_hook(hook))

    def record(self, key, tensors):
        samples = [
            (tensor.detach().float().square().sum(), tensor.numel())
            for tensor in tensors
            if isinstance(tensor, torch.Tensor) and tensor.numel()
        ]
        if samples:
            self.samples.setdefault(key, []).extend(samples)
        return samples

    def __call__(self, ctx):
        # Arm hooks for the next optimizer step, including all its microbatches.
        self.capture = (ctx.step + 1) % self.every == 0
        if ctx.step % self.every:
            return {}

        for name, module in ctx.model.named_modules():
            params = list(module.parameters(recurse=False))
            prefix = f"rms/{name or 'root'}"
            self.record(f"{prefix}/parameter", params)
            self.record(f"{prefix}/gradient", [p.grad for p in params])

        params = list(ctx.model.parameters())
        self.record("rms/global/parameter", params)
        self.record("rms/global/gradient", [p.grad for p in params])
        stats = {
            key: (sum(s for s, n in samples) / sum(n for s, n in samples)).sqrt()
            for key, samples in self.samples.items()
        }
        self.samples.clear()
        return stats

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.samples.clear()
        self.capture = False


log_module_rms = ModuleRMSLogger(every=100)
log_module_rms.__name__ = log_module_rms.__qualname__ = "log_module_rms"
