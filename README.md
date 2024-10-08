# Tiny LLM

## Introduction

**Tiny LLM** is a small language model built from scratch using PyTorch. This project was created for educational purposes, inspired by GPT-2 and GPT-3 architectures.

The model consists of:
- **12 layers** with **12 attention heads**
- **~125 million parameters**

### Model Performance

Tiny LLM is capable of generating coherent text based on previously inputted tokens. The model was trained on the **fineweb-edu** dataset, comprising 10 billion tokens (approximately 20 GB in size). After training for a single epoch, it achieved the following results:

- **Validation Loss**: `2.952`  
  *GPT-2 baseline*: `3.292`
- **HellaSwag Test Accuracy**: `0.299`  
  *GPT-2 baseline*: `0.294`

The model file size is approximately 1.5 GB.

## Acknowledgments

This project was made possible with inspiration and resources from the following sources:

- **[GPT-3 Research Paper (OpenAI)](https://arxiv.org/abs/2005.14165)**  
  Comprehensive paper on GPT-3's architecture, methodology, and benchmarks.

- **[NanoGPT by Andrej Karpathy](https://github.com/karpathy/nanoGPT)**  
  A minimal implementation of GPT, along with insightful [YouTube tutorials](https://www.youtube.com/watch?v=kCc8FmEb1nY).

- **[fineweb-edu Dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)**  
  The dataset used for training Tiny LLM, hosted on Hugging Face.
