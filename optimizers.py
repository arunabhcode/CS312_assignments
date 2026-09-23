import torch


NORM_CLASS_NAME_FRAGMENTS = ("LayerNorm", "RMSNorm", "RmsNorm")
ADAMW_EPSILON = 1e-8


def is_norm_module(module):
    class_name = module.__class__.__name__
    return any(fragment in class_name for fragment in NORM_CLASS_NAME_FRAGMENTS)


def should_apply_weight_decay(module, parameter_name):
    if isinstance(module, torch.nn.Embedding):
        return False
    if is_norm_module(module):
        return False
    if parameter_name.endswith("bias"):
        return False
    return True


def build_masked_weight_decay_parameter_groups(model, weight_decay):
    decayed_parameters = []
    non_decayed_parameters = []
    seen_parameter_ids = set()

    for module in model.modules():
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            parameter_id = id(parameter)
            if parameter_id in seen_parameter_ids:
                continue
            seen_parameter_ids.add(parameter_id)

            if should_apply_weight_decay(module, parameter_name):
                decayed_parameters.append(parameter)
            else:
                non_decayed_parameters.append(parameter)

    parameter_groups = []
    if decayed_parameters:
        parameter_groups.append({"params": decayed_parameters, "weight_decay": weight_decay})
    if non_decayed_parameters:
        parameter_groups.append({"params": non_decayed_parameters, "weight_decay": 0.0})

    return parameter_groups


def build_optimizer(
    model,
    optimizer_name,
    learning_rate,
    weight_decay,
    beta1,
    beta2,
):
    if optimizer_name not in {"adamw", "sgd"}:
        raise ValueError(f"Unsupported optimizer {optimizer_name!r}; expected 'adamw' or 'sgd'.")

    optimizer_parameters = build_masked_weight_decay_parameter_groups(
        model,
        weight_decay,
    )
    if optimizer_name == "adamw":
        optimizer_kwargs = {
            "lr": learning_rate,
            "weight_decay": 0.0,
            "betas": (beta1, beta2),
            "eps": ADAMW_EPSILON,
        }
        if any(
            parameter.is_cuda
            for group in optimizer_parameters
            for parameter in group["params"]
        ):
            optimizer_kwargs["fused"] = True
        return torch.optim.AdamW(optimizer_parameters, **optimizer_kwargs)

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            optimizer_parameters,
            lr=learning_rate,
            momentum=beta1,
            weight_decay=0.0,
        )
