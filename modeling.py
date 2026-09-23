import math

import torch
from torch import nn
from torch.nn import functional as F

from model_config import LMConfig


RMS_NORM_EPS = 1e-5
QK_NORM_EPS = 1e-5
ROPE_THETA = 500_000.0


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        inv_freq = 1.0 / (
            ROPE_THETA
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, hidden_states, position_ids):
        inv_freq = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(hidden_states.device)
        )
        position_ids = position_ids[:, None, :].float()
        device_type = hidden_states.device.type
        if not isinstance(device_type, str) or device_type == "mps":
            device_type = "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq.float() @ position_ids.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(hidden_states.dtype), sin.to(hidden_states.dtype)


def rotate_half(hidden_states):
    first_half = hidden_states[..., : hidden_states.shape[-1] // 2]
    second_half = hidden_states[..., hidden_states.shape[-1] // 2 :]
    return torch.cat((-second_half, first_half), dim=-1)


def apply_rotary_pos_emb(query_states, key_states, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    query_states = (query_states * cos) + (rotate_half(query_states) * sin)
    key_states = (key_states * cos) + (rotate_half(key_states) * sin)
    return query_states, key_states


def repeat_kv(hidden_states, num_repeats):
    if num_repeats == 1:
        return hidden_states
    batch_size, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch_size,
        num_key_value_heads,
        num_repeats,
        seq_len,
        head_dim,
    )
    return hidden_states.reshape(batch_size, num_key_value_heads * num_repeats, seq_len, head_dim)


def causal_attention_mask(seq_len, attention_mask, device, dtype):
    min_dtype = torch.finfo(dtype).min
    mask = torch.full((seq_len, seq_len), min_dtype, device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    mask = mask[None, None, :, :]
    if attention_mask is None:
        return mask

    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must have shape [batch, seq], got {attention_mask.shape}.")
    padding_mask = attention_mask[:, None, None, :].to(torch.bool)
    return mask.masked_fill(~padding_mask, min_dtype)


class LlamaAttention(nn.Module):
    def __init__(self, config, qk_norm=True, dropout=0.0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.qk_norm = bool(qk_norm)
        self.dropout = float(dropout)

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )
        if self.qk_norm:
            self.q_norm = LlamaRMSNorm(self.head_dim, eps=QK_NORM_EPS)
            self.k_norm = LlamaRMSNorm(self.head_dim, eps=QK_NORM_EPS)

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size,
            seq_len,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size,
            seq_len,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if self.qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is None:
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_attention_mask(
                    seq_len,
                    attention_mask,
                    hidden_states.device,
                    query_states.dtype,
                ),
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states):
        hidden_states = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(hidden_states)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx, qk_norm=True, dropout=0.0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dropout = float(dropout)
        self.self_attn = LlamaAttention(config, qk_norm=qk_norm, dropout=self.dropout)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=RMS_NORM_EPS)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=RMS_NORM_EPS)

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        return residual + hidden_states


class LlamaModel(nn.Module):
    def __init__(self, config, qk_norm=True, dropout=0.0):
        super().__init__()
        self.config = config
        self.qk_norm = bool(qk_norm)
        self.dropout = float(dropout)
        self.padding_idx = None
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(
                    config,
                    layer_idx,
                    qk_norm=self.qk_norm,
                    dropout=self.dropout,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=RMS_NORM_EPS)
        self.rotary_emb = LlamaRotaryEmbedding(config)
        self.gradient_checkpointing = False

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [batch, seq], got {input_ids.shape}.")
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                "Sequence length "
                f"{seq_len} exceeds max_position_embeddings="
                f"{self.config.max_position_embeddings}."
            )

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(
            hidden_states,
            position_ids,
        )

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )
        return self.norm(hidden_states)


class AutoregressiveLM(nn.Module):
    def __init__(
        self,
        config,
        dtype=torch.float32,
        qk_norm=True,
        tie_word_embeddings=False,
        dropout=0.0,
    ):
        super().__init__()
        if not isinstance(config, LMConfig):
            config = LMConfig.from_dict(config)
        self.config = config
        self.qk_norm = bool(qk_norm)
        self.tie_word_embeddings = bool(tie_word_embeddings)
        self.dropout = float(dropout)
        self.model = LlamaModel(config, qk_norm=self.qk_norm, dropout=self.dropout)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if self.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.to(dtype=dtype)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None):
        if input_ids is None:
            raise ValueError("input_ids must be provided.")
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return self.lm_head(hidden_states)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def _truncated_normal_(tensor, scale):
    nn.init.trunc_normal_(tensor, mean=0.0, std=scale, a=-3.0 * scale, b=3.0 * scale)


def initialize_model(model):
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()

    skipped_modules = {id(input_embeddings), id(output_embeddings)}

    with torch.no_grad():
        for module in model.modules():
            if id(module) in skipped_modules:
                continue
            if isinstance(module, nn.Linear):
                fan_in = module.weight.shape[1]
                _truncated_normal_(module.weight, 1.0 / math.sqrt(fan_in))
                if module.bias is not None:
                    module.bias.zero_()
            elif isinstance(module, LlamaRMSNorm):
                module.weight.fill_(1.0)

        hidden_size = input_embeddings.weight.shape[1]
        base = torch.empty_like(input_embeddings.weight, dtype=torch.float32)
        _truncated_normal_(base, 1.0)
        base = base.to(dtype=input_embeddings.weight.dtype)
        if getattr(model, "tie_word_embeddings", False):
            input_embeddings.weight.copy_(base / math.sqrt(hidden_size))
            return

        input_embeddings.weight.copy_(base / hidden_size)
        if output_embeddings.weight.shape == input_embeddings.weight.shape:
            output_embeddings.weight.copy_(base / math.sqrt(hidden_size))
        else:
            _truncated_normal_(output_embeddings.weight, 1.0 / math.sqrt(hidden_size))
        if output_embeddings.bias is not None:
            output_embeddings.bias.zero_()
