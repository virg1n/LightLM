# Tiny LLM - 124M Parameter Language Model

**Tiny LLM (124M)** is a small-scale language model inspired by GPT-2 and GPT-3 architectures. 

### Model Architecture
- **12 transformer layers** with **12 attention heads**
- **~124 million parameters**
- **bfloat16 precision** for faster training
- **Distributed Data Parallel (DDP)** for efficient multi-GPU training

### Dataset

The model was trained on the **fineweb-edu** dataset, comprising **10 billion tokens** (~30x smaller than GPT-3's training data). The training data was preprocessed and tokenized in a format similar to GPT-style models.

### Performance
Tiny LLM achieved competitive results using a much smaller dataset and fewer resources:

- **Validation Loss**: `2.85`
- **HellaSwag Test Accuracy**: `33%`  
  *(Comparable to GPT-3's performance on a similar 124M model)*

The model is trained for **100k steps**, and we have visualizations of training and validation loss over time, which you can find below (images included).

---

### Model Demo

The model is deployed and accessible at the following link:  
**[Tiny LLM Demo](https://lm.vviky.com)**

Here is an example of the output when prompted with: "Hello, I am a language model,":

```
Hello, I am a language model, inspired by the delightful Chinese philosophers, Bo ibidatus, and of course
Hello, I am a language model, a boolean module, and have a vision of manipulating data in C
Hello, I am a language model, and this topic is important for Blockchain. This means that it will be important
Hello, I am a language model, programmer, and consultant who works at the Unity language team at OMG (Internet
Hello, I am a language model, university teacher, teacher and administrator. In addition to teaching, I write articles
```

---

### Training and Loss Progress

Over the course of ~100k steps, the model's training loss and validation loss showed the following trends :
[TODO] add images

---

### Acknowledgments

This project was made possible with the inspiration and knowledge provided by the following sources:

- **[GPT-3 Research Paper (OpenAI)](https://arxiv.org/abs/2005.14165)**  
Comprehensive paper detailing GPT-3's architecture, training methodology, and benchmark performance.

- **[NanoGPT by Andrej Karpathy](https://github.com/karpathy/nanoGPT)**  
A minimal implementation of GPT that served as a practical guide for this project.

- **[fineweb-edu Dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)**  
  The dataset used for training Tiny LLM, hosted on Hugging Face.

- ...
