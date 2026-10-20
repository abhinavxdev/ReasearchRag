# Live two-paper demo

Real arXiv PDFs, BGE embeddings, Chroma/BM25/RRF, and Ollama. See verification.md for model and execution details.

## Which optimizer did the Transformer use?

We used the Adam optimizer (20) with β1 = 0.9, β2 = 0.98 and ε = 10−9 [Attention Is All You Need, p.7, Optimizer].

### Source: Attention Is All You Need, p.7, Optimizer
ID: `S_4d5450091387acb393f07768`

```text
We used the Adam optimizer [20] with β1 = 0.9, β2 = 0.98 and ε = 10−9. We varied the learning
rate over the course of training, according to the formula:

             lrate = d−0.5model · min(step_num−0.5, step_num · warmup_steps−1.5)           (3)

This corresponds to increasing the learning rate linearly for the first warmup_steps training steps,
and decreasing it thereafter proportionally to the inverse square root of the step number. We used
warmup_steps = 4000.
```

## How many learning rate warmup steps were used?

We used warmup_steps = 4000 [Attention Is All You Need, p.7, Optimizer].

### Source: Attention Is All You Need, p.7, Optimizer
ID: `S_4d5450091387acb393f07768`

```text
We used the Adam optimizer [20] with β1 = 0.9, β2 = 0.98 and ε = 10−9. We varied the learning
rate over the course of training, according to the formula:

             lrate = d−0.5model · min(step_num−0.5, step_num · warmup_steps−1.5)           (3)

This corresponds to increasing the learning rate linearly for the first warmup_steps training steps,
and decreasing it thereafter proportionally to the inverse square root of the step number. We used
warmup_steps = 4000.
```

## What percentage of WordPiece tokens is masked during BERT pre-training?

In all of our experiments, we mask 15% of all WordPiece tokens in each sequence at random [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, p.4, Pre-Training Bert].

### Source: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, p.4, Pre-Training Bert
ID: `S_a2e2fc24e72189b57db31786`

```text
Unlike Peters et al. (2018a) and Radford et al.
(2018), we do not use traditional left-to-right or
right-to-left language models to pre-train BERT.
Instead, we pre-train BERT using two unsupervised tasks, described in this section. This step
is presented in the left part of Figure 1.

Task #1: Masked LM
Intuitively, it is reasonable to believe that a deep bidirectional model is
strictly more powerful than either a left-to-right
model or the shallow concatenation of a left-toright and a right-to-left model.
Unfortunately,
standard conditional language models can only be
trained left-to-right or right-to-left, since bidirectional conditioning would allow each word to indirectly “see itself”, and the model could trivially
predict the target word in a multi-layered context.

former is often referred to as a “Transformer encoder” while
the left-context-only version is referred to as a “Transformer
decoder” since it can be used for text generation.

In order to train a deep bidirectional representation, we simply mask some percentage of the input
tokens at random, and then predict those masked
tokens. We refer to this procedure as a “masked
LM” (MLM), although it is often referred to as a
Cloze task in the literature (Taylor, 1953). In this
case, the final hidden vectors corresponding to the
mask tokens are fed into an output softmax over
the vocabulary, as in a standard LM. In all of our
experiments, we mask 15% of all WordPiece tokens in each sequence at random. In contrast to
denoising auto-encoders (Vincent et al., 2008), we
only predict the masked words rather than reconstructing the entire input.

Although this allows us to obtain a bidirectional pre-trained model, a downside is that we
are creating a mismatch between pre-training and
fine-tuning, since the [MASK] token does not appear during fine-tuning. To mitigate this, we do
not always replace “masked” words with the actual [MASK] token. The training data generator
chooses 15% of the token positions at random for
prediction. If the i-th token is chosen, we replace
the i-th token with (1) the [MASK] token 80% of
the time (2) a random token 10% of the time (3)
the unchanged i-th token 10% of the time. Then,
Ti will be used to
```
