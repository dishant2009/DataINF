# DataInf: Efficient Estimation of Data Influence in LoRA-tuned LLMs and Diffusion Models

This repository provides a Python implementation of **DataInf**, the method introduced in  
[DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models (ICLR 2024)](https://arxiv.org/abs/2310.00902).  

DataInf is a closed-form influence approximation method that efficiently identifies the most (or least) influential training data points in **large language models (LLMs)** and **diffusion models**, especially when fine-tuned with parameter-efficient methods like **LoRA**.

---

## Key Features
- **Closed-form influence approximation** without expensive iterative solvers.
- Scales to **large-scale LLMs** (e.g., RoBERTa, LLaMA-2) and **diffusion models**.
- Identifies **mislabeled or harmful data points** in fine-tuning datasets.
- **Orders of magnitude faster** than traditional influence function estimation methods such as LiSSA.

---

## Paper Summary
- Traditional influence functions require **Hessian inversion**, which is intractable for large models.
- DataInf leverages a **Sherman-Morrison-based approximation** to swap matrix inversion and averaging, yielding a **fast closed-form solution**.
- Empirically validated on:
  - **RoBERTa-large** for GLUE benchmarks (error analysis, mislabeled data detection).
  - **LLaMA-2-13B-chat** for text generation.
  - **Stable Diffusion v1.5** for text-to-image generation.
- Outperforms baselines (LiSSA, Hessian-free) in both **accuracy** and **runtime**:contentReference[oaicite:1]{index=1}.

---

## Repository Structure
- `DataINF.py` – Core implementation of the DataInf algorithm.
- `DataINF.pdf` – The ICLR 2024 paper describing the method in detail.

---

