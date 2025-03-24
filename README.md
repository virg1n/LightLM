# LightLM - <150M Parameter Language Model

**LightLM** is a language model with up to 150M parameters.
This repository explores the limits of small language models, pushing how smart they can be. LightLM integrates the latest architectural innovations and dataset improvements to enhance the coherence of its output. 

## Current LightLM
### Model Architecture
- **~30 transformer layers**
- **Grouped Querry Attention**
- **FeedForward Layers With SwiGLU**
- **Rotary Position Embedding (RoPE)**
- **KV-cache**
- **RMSNormalization**
- **Optional Mixture of Experts (MoE)**
- **Loss-free Load balancing and DeepSeekMoE**

### Dataset
The model was trained on the [Cosmopedia v2](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) dataset (~28 billion tokens). 

### Performance
```ARC-C Accuracy: 27.2% 
WinoGrande Accuracy: 52.8%```


Here is example of the output when prompted with: "Hello, I am a language model,":
```
Hello, I am a language model, and I can help you learn more about the language you are interested in. Let's start with the basics.

Hello, I am a language model, and I can help you learn some new words and phrases. Maybe you could try saying "hello" in English first, then move on to Spanish,
```

You can download weights [here](https://huggingface.co/Virg1n/LightLM).

### Acknowledgments

This project was made possible with the inspiration and knowledge provided by the following sources:

- **[GPT-3 Research Paper (OpenAI)](https://arxiv.org/abs/2005.14165)**  

- **[NanoGPT by Andrej Karpathy](https://github.com/karpathy/nanoGPT)**  

- **[fineweb-edu Dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)**  

- **[MobileLLM](https://arxiv.org/pdf/2402.14905)**

https://arxiv.org/pdf/2412.19437
https://github.com/meta-llama/llama
