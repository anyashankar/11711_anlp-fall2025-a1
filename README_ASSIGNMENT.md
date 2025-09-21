# Assignment 1 – Build Your Own LLaMA

**Course:** 11-711 / 11-611 Advanced NLP (Fall 2025)  
**Due:** September 25, 2025 at 11:59 PM  
**Student:** Anya Shankar  

---

## Overview

In this assignment, I implemented core components of the **LLaMA2 transformer architecture** and applied the model to text generation and sentiment classification tasks.  
Implemented components include:

- **LayerNorm** (without bias)  
- **Scaled Dot-Product Attention** with Grouped Query Attention (GQA)  
- **Rotary Positional Embeddings (RoPE)**  
- **LoRA (Low-Rank Adaptation)**  
- **AdamW Optimizer** with gradient clipping and weight decay  
- **Epsilon Sampling** for text generation  
- **Classifier** for sentiment analysis  

Key fix: Removed incorrect causal masking in attention → sanity check passed.  

---

## Results

- **Sanity Check:** All outputs match expected values  
- **RoPE Test:** Correct implementation verified  
- **Optimizer Test:** AdamW functions correctly  
- **Text Generation:** Produces coherent continuations at multiple temperatures  
- **Zero-shot Classification:** 51.8% dev accuracy  
- **Fine-tuning:** 80.7% train / 80.0% dev accuracy (after 1 epoch)  
- **LoRA Fine-tuning:** Only **0.39% parameters** trained, with full model capability maintained  

---

## References

- Assignment instructions: [CMU ANLP Assignment 1](https://cmu-l3.github.io/anlp-fall2025/assignments/assignment1)  
- Ba et al., 2016 — *Layer Normalization*  
- Vaswani et al., 2017 — *Attention Is All You Need*  
- Su et al., 2021 — *RoFormer: Rotary Position Embeddings*  
- Ainslie et al., 2023 — *Grouped Query Attention*  
- Hu et al., 2021 — *LoRA: Low-Rank Adaptation*  
- Kingma & Ba, 2014; Loshchilov & Hutter, 2017 — *Adam / AdamW*  
- Hewitt et al., 2022 — *Truncation Sampling*  
- Socher et al., 2013 — *Recursive Deep Models for Sentiment* 

---
