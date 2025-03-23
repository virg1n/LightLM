# LightLM - <150M Parameter Language Model

**LightLM** is a language model with up to 150M parameters.
This repository explores the limits of small language models, pushing how smart they can be. LightLM was initially constructed with GPT-2 architecture from the instructions in Karpathy's tutorial. Now, it integrates the latest architectural innovations and dataset improvements to enhance the coherence of its output. The goal is to compare its performance against GPT-2 124M and other small language models, highlighting the benefits of these advancements.

## Current LightLM
### Model Architecture
- **30 transformer layers**
- **Grouped Querry Attention**
- **FeedForward Layers With SwiGLU**
- **Rotary Position Embedding (RoPE)**
- **KV-cache**
- **RMSNormalization**
- **Optional Mixture of Experts (MoE)**
- **Loss-free Load balancing and DeepSeekMoE**

### Dataset
The model will be trained on the **fineweb-edu** dataset

### Performance
Soon...


## Initial LightLM/GPT-2 (124M)
### Model Architecture
- **12 transformer layers** with **12 attention heads**
- **Self-Attention**
- **FeedForward Layers With GeLU**

### Dataset
The model was trained on the **fineweb-edu** dataset with 4 epochs (**40 billion tokens**). 
Tokenizer: GPT-2 Tokemizer

### Performance
- **Validation Loss**: `2.85`
- **HellaSwag Test Accuracy**: `33%`  

Here is an example of the output when prompted with: "Hello, I am a language model,":

```
Hello, I am a language model, inspired by the delightful Chinese philosophers, Bo ibidatus, and of course
Hello, I am a language model, programmer, and consultant who works at the Unity language team at OMG
```



### Acknowledgments

This project was made possible with the inspiration and knowledge provided by the following sources:

- **[GPT-3 Research Paper (OpenAI)](https://arxiv.org/abs/2005.14165)**  

- **[NanoGPT by Andrej Karpathy](https://github.com/karpathy/nanoGPT)**  

- **[fineweb-edu Dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)**  

- **[MobileLLM](https://arxiv.org/pdf/2402.14905)**

https://arxiv.org/pdf/2412.19437
https://github.com/meta-llama/llama
