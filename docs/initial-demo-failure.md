# Live two-paper demo

Provider and models are configured via environment.

## Which optimizer did the Transformer use?

The optimizer was Adam [Attention Is All You Need, p.7, Optimizer].

### Source: Attention Is All You Need, p.7, Optimizer
ID: `S_8ca676585321ad4a42d16482`

```text
We used the Adam optimizer [20] with β1 = 0.9, β2 = 0.98 and ϵ = 10−9. We varied the learning
rate over the course of training, according to the formula:

             lrate = d−0.5model · min(step_num−0.5, step_num · warmup_steps−1.5)           (3)

This corresponds to increasing the learning rate linearly for the first warmup_steps training steps,
and decreasing it thereafter proportionally to the inverse square root of the step number. We used
warmup_steps = 4000.
```

## How many learning rate warmup steps were used?

warmup_steps = 4000 [Attention Is All You Need, p.7, Optimizer].

### Source: Attention Is All You Need, p.7, Optimizer
ID: `S_8ca676585321ad4a42d16482`

```text
We used the Adam optimizer [20] with β1 = 0.9, β2 = 0.98 and ϵ = 10−9. We varied the learning
rate over the course of training, according to the formula:

             lrate = d−0.5model · min(step_num−0.5, step_num · warmup_steps−1.5)           (3)

This corresponds to increasing the learning rate linearly for the first warmup_steps training steps,
and decreasing it thereafter proportionally to the inverse square root of the step number. We used
warmup_steps = 4000.
```

## What percentage of WordPiece tokens is masked during BERT pre-training?

The tokenizer was BertTokenizer [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, p.4, Introduction].

### Source: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, p.4, Introduction
ID: `S_69c58239781b803431e21dfd`

```text
Input/Output Representations  To make BERT      In order to train a deep bidirectional representahandle a variety of down-stream tasks, our input    tion, we simply mask some percentage of the input
representation is able to unambiguously represent    tokens at random, and then predict those masked
both a single sentence and a pair of sentences    tokens. We refer to this procedure as a “masked
(e.g., ⟨Question, Answer ⟩) in one token sequence.  LM” (MLM), although it is often referred to as a
Throughout this work, a “sentence” can be an arbi-   Cloze task in the literature (Taylor, 1953). In this
trary span of contiguous text, rather than an actual    case, the ﬁnal hidden vectors corresponding to the
linguistic sentence. A “sequence” refers to the in-   mask tokens are fed into an output softmax over
put token sequence to BERT, which may be a sin-    the vocabulary, as in a standard LM. In all of our
gle sentence or two sentences packed together.       experiments, we mask 15% of all WordPiece to-
  We use WordPiece embeddings (Wu  et  al.,   kens in each sequence at random. In contrast to
2016) with a 30,000 token vocabulary. The ﬁrst    denoising auto-encoders (Vincent et al., 2008), we
token of every sequence is always a special clas-    only predict the masked words rather than reconsiﬁcation token ([CLS]). The ﬁnal hidden state    structing the entire input.
corresponding to this token  is used as the ag-     Although this allows us to obtain a bidirecgregate sequence representation for classiﬁcation    tional pre-trained model, a downside is that we
tasks. Sentence pairs are packed together into a    are creating a mismatch between pre-training and
single sequence. We differentiate the sentences in    ﬁne-tuning, since the [MASK] token does not aptwo ways. First, we separate them with a special    pear during ﬁne-tuning. To mitigate this, we do
token ([SEP]). Second, we add a learned embed-    not always replace “masked” words with the acding to every token indicating whether it belongs    tual [MASK] token. The training data generator
to sentence A or sentence B. As shown in Figure 1,   chooses 15% of the token positions at random for
we denote input embedding as E, the ﬁnal hidden    prediction. If the
```
