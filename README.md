# BitNet-1.58-Instruct

Implementation of BitNet-1.58 instruct tuning. This work builds off the pre-trained models released in the [1bitLLM/bitnet_b1_58-large](https://huggingface.co/1bitLLM/bitnet_b1_58-large) project on Hugging Face.

## Inference

To simply prompt the model in a loop, you can run this script:

```bash
TODO:
```

## Pre-Training

The models are trained with [RedPajama dataset](https://github.com/togethercomputer/RedPajama-Data) for 100B tokens. The hypers, as well as two-stage LR and weight decay, are implemented as suggested in their following paper. 

NOTE: This repo does not perform the pre-training, just uses these models as a jumping off point for instruct tuning.

## Instruct Tuning

### Data

The instruct tuning was done on a mix of data:

1) SQuADv2 with context and questions
2) mosaicml/instruct-v3

You can see the mix of data here:

https://www.oxen.ai/ox/BitNet/file/main/train.jsonl

```bash
oxen download ox/BitNet train.jsonl
oxen download ox/BitNet dev.jsonl
```

### Code

The training was done on an A10 with 24GB of VRAM. We cut off the max seq len to 768 because otherwise it runs out of VRAM on some batches. Would be nice to kick off a train on a larger GPU and larger context length.

```bash
python tools/train.py -d -m 1bitLLM/bitnet_b1_58-large -d train.jsonl -o results/bitnet_b1_58-large-instruct --max_seq_len 768
```

## Evaluation

For evaluation purposes, we are also using SQuAD dataset. The idea is the model should be able to answer generic questions as well as extract answers from questions and context if provided.

If the answer is not in the context, we want to be able to say "Not in context.".

```bash
python tools/eval.py -m results/bitnet_b1_58-large-instruct/final_checkpoint/ -d dev.jsonl -o eval.jsonl -n 100
```

The eval script outputs a dataframe like this:

