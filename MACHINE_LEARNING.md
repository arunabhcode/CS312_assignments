# Machine Learning Handout

This note describes the learning problem, model, data, and training recipe used
in this repo. It is meant to help you understand what a run is optimizing. For
Modal setup and commands, use [README.md](README.md).

## Task

The task is autoregressive language modeling. Each training example is a fixed
length sequence of token IDs:

```text
x = [x_0, x_1, ..., x_1023]
```

The model reads the prefix up to each position and predicts the next token. At
position `t`, the prediction target is `x_{t+1}`. The first token in each packed
sequence is a beginning-of-sequence token.

The core object of study is the next-token training loss. Sampling and
generation are not part of the default training exercise.

## Tokenization And Data

The default data is a tokenized DCLM subset using the course SentencePiece
tokenizer:

```text
tokenizer: parameter-golf-sp4096
vocab size: 4096
context length: 1024
```

Important special token IDs:

```text
pad: 0
bos: 1
eos: 2
unk: 3
```

The training split contains `9,600,000` token sequences. The default run uses
the first `600,000` sequences after applying the default data ordering. This is:

```text
600,000 sequences * 1024 tokens/sequence = 614,400,000 tokens
```

The validation split contains `1,000` sequences. Validation uses the same
sequence length and tokenizer as training.

The default data seed is `42`. A non-`None` data seed means the training rows are
globally permuted once, and the run takes the requested prefix of that global
permutation. For `N` training sequences, the effective data is:

```text
base_train[permutation(seed)[:N]]
```

Training then reads the selected rows sequentially. The validation set is not
shuffled by the data seed.

## Model Family

The built-in models are named by depth:

```text
d4, d5, d6, ..., d18
```

For a model named `dX`:

```text
num layers: X
hidden size: 64 * X
attention head dimension: 64
num attention heads: hidden size / 64 = X
MLP intermediate size: floor(3.5 * hidden size)
vocab size: 4096
max position embeddings: 131,072
```

The default model is `d8`, so it has 8 transformer blocks, hidden size 512, and
8 attention heads.
These architectural quantities are fixed for the built-in depth ladder. To
change width, MLP size, head layout, vocabulary size, context length, or related
architecture details, pass a custom `LMConfig` through `TrainConfig.model_config`.

## Transformer Block

The model is a decoder-only transformer. Each sequence is processed by:

```text
token embedding
decoder block 1
decoder block 2
...
decoder block L
final RMSNorm
linear LM head
```

The model outputs logits with shape:

```text
batch_size x sequence_length x vocab_size
```

Each decoder block uses a pre-norm residual structure:

```text
h = h + attention(RMSNorm(h))
h = h + mlp(RMSNorm(h))
```

By default, dropout is zero. Setting `TrainConfig.dropout` applies the same
dropout probability to attention weights and to the attention/MLP branch outputs
before each residual addition.

## Attention

Attention is causal: token position `t` can attend only to positions `<= t`.

The attention projections are:

```text
Q = h W_q
K = h W_k
V = h W_v
```

The built-in models use the same number of query, key, and value heads. That is
standard multi-head attention, not grouped-query attention.

Rotary position embeddings are applied to queries and keys. The RoPE base is:

```text
theta = 500,000
```

After RoPE, query-key normalization is enabled by default. For each query and
key head vector, RMSNorm is applied over the head dimension before the attention
dot products are computed. This is controlled by `qk_norm`; the default is
`True`.

Attention probabilities are not explicitly formed in the handout notation, but
the operation is the usual scaled dot-product attention:

```text
softmax(Q K^T / sqrt(head_dim) + causal_mask) V
```

## Normalization And MLP

RMSNorm normalizes each hidden vector by its root mean square:

```text
RMSNorm(x) = gamma * x / sqrt(mean(x^2) + eps)
```

The default epsilon is:

```text
eps = 1e-5
```

The MLP is SwiGLU:

```text
MLP(x) = W_down(silu(W_gate x) * W_up x)
```

All linear layers in the transformer and MLP are bias-free in the default
architecture.

## Loss

For a batch of token sequences, logits at the final position are dropped because
there is no next token inside the sequence. Labels at the first position are
dropped because nothing predicts the first token.

```text
logits_for_loss = logits[:, :-1, :]
labels_for_loss = input_ids[:, 1:]
```

The training loss is mean next-token negative log-likelihood:

```text
loss = -mean(log p_model(labels_for_loss | previous tokens))
```

Equivalently, for batch size `B` and sequence length `T = 1024`:

```text
loss = -1 / (B * (T - 1)) * sum_{b=1}^B sum_{t=0}^{T-2}
       log p_model(x_{b,t+1} | x_{b,0:t})
```

Validation reports the same token-averaged negative log-likelihood on held-out
sequences.

## Initialization

The default initialization is part of the training recipe.

For ordinary linear layers:

```text
weight ~ truncated_normal(mean=0, std=1/sqrt(fan_in), limits=+/-3 std)
```

For RMSNorm:

```text
gamma = 1
```

For token embeddings:

```text
embedding weight ~ truncated_normal(0, 1) / hidden_size
```

For the LM head, when its shape matches the embedding matrix:

```text
LM head weight uses the same base sample scaled by 1/sqrt(hidden_size)
```

This means input embeddings and output logits start from related directions but
with different scaling.

By default, the input embedding matrix and LM-head matrix are separate
parameters. Setting `tie_word_embeddings=True` shares one matrix between them,
so the tied matrix uses the LM-head scale above.

## Default Training Recipe

The default reference recipe is:

```text
model: d8
train sequences: 600,000
validation sequences: 1,000
batch size: 64
micro-batches: 1
epochs: 1.0
optimizer: AdamW
learning rate: 0.003
Adam beta1: 0.9
Adam beta2: 0.95
weight decay: 0.1
gradient clipping: global norm 1.0
tie word embeddings: false
learning-rate schedule: linear decay
warmup fraction: 0.01
precision: mixed precision
data seed: 42
model seed: 42
deterministic training: false
qk norm: enabled
```

With `600,000` sequences and batch size `64`, one epoch is:

```text
floor(600,000 / 64) = 9,375 optimizer steps
```

The final partial batch is not used.

## Optimizer

The default optimizer is AdamW. SGD is also supported for experiments.

Weight decay is masked:

```text
decay: ordinary linear weights
no decay: token embeddings, normalization weights, bias terms
```

The AdamW update uses the configured learning rate, betas, masked weight decay,
and a fixed epsilon:

```text
1e-8
```

## Learning Rate Schedule

The default schedule is warmup followed by linear decay to zero.

Warmup steps:

```text
warmup_steps = int(total_steps * warmup_percent)
```

During warmup, the learning rate increases linearly from 0 to the configured
learning rate. After warmup, it decays linearly to 0 by the final optimizer
step.

Other schedules can be useful for experiments, but linear decay is the default
reference recipe.

## Precision

The default precision mode is mixed precision, abbreviated as `mp`:

```text
parameters: fp32
forward/backward compute on CUDA: bf16 autocast
loss computation: fp32
optimizer state: fp32
```

The model can also run in pure `fp32` or pure `bf16`, but `mp` is the default
training recipe.

## Evaluation

The run evaluates periodically throughout training. The default number of
evaluation points is `100`, so evaluation is frequent enough to show the shape
of the learning curve.

Validation loss is token-weighted across validation batches. This matters
because the reported validation loss should be an average over predicted tokens,
not an average over batches.

## Checkpoints

A completed run produces a final model artifact containing:

```text
model configuration
model weights
run metadata
```

During training, resumable checkpoints can also store:

```text
model state
optimizer state
scheduler state through the completed step count
random number generator state
completed optimizer step
W&B run id
run metadata
```

The default latest checkpoint frequency is every `1,000` optimizer steps, plus
the final step. A latest checkpoint is overwritten as training progresses. You
can also request kept step checkpoints for specific optimizer steps when you
want to branch later from an intermediate model.

Loading a final model starts from model weights only. Resuming a run from its
latest training checkpoint restores the optimizer, learning-rate schedule,
random number generators, and W&B run identity.
