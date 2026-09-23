from dataclasses import asdict, dataclass

import torch


VALID_PRECISIONS = {"fp32", "bf16", "mp"}
DEFAULT_VOCAB_SIZE = 4096
DEFAULT_CONTEXT_LENGTH = 1024
DEFAULT_MAX_POSITION_EMBEDDINGS = 131_072
DEPTHS = range(4, 19)


@dataclass(frozen=True)
class LMConfig:
    name: str
    vocab_size: int
    context_length: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int = 64
    max_position_embeddings: int = DEFAULT_MAX_POSITION_EMBEDDINGS

    def __post_init__(self):
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError(
                "hidden_size must equal num_attention_heads * head_dim: "
                f"{self.hidden_size} != {self.num_attention_heads} * {self.head_dim}"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads: "
                f"{self.num_attention_heads} % {self.num_key_value_heads}"
            )
        if self.context_length > self.max_position_embeddings:
            raise ValueError(
                "context_length cannot exceed max_position_embeddings: "
                f"{self.context_length} > {self.max_position_embeddings}"
            )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        fields = cls.__dataclass_fields__
        return cls(**{key: data[key] for key in fields if key in data})


def model_dtype_for_precision(precision):
    validate_precision(precision)
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


def validate_precision(precision):
    if precision not in VALID_PRECISIONS:
        raise ValueError(
            f"Invalid precision={precision!r}. Expected one of {sorted(VALID_PRECISIONS)}."
        )


def depth_model_config(depth, vocab_size=DEFAULT_VOCAB_SIZE, context_length=DEFAULT_CONTEXT_LENGTH):
    hidden_size = 64 * depth
    return LMConfig(
        name=f"d{depth}",
        vocab_size=vocab_size,
        context_length=context_length,
        hidden_size=hidden_size,
        intermediate_size=int(hidden_size * 3.5),
        num_hidden_layers=depth,
        num_attention_heads=hidden_size // 64,
        num_key_value_heads=hidden_size // 64,
        head_dim=64,
    )


MODEL_CONFIGS = {f"d{depth}": depth_model_config(depth) for depth in DEPTHS}


def resolve_model_config(
    model_name=None,
    model_config=None,
):
    if model_config is not None:
        return model_config
    if model_name is None:
        raise ValueError("Either model_name or model_config must be provided.")
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_name={model_name!r}. Expected {sorted(MODEL_CONFIGS)}.")
    return MODEL_CONFIGS[model_name]
