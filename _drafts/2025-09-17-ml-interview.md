---
layout: post
title: ML Interview Questions List
date: 2025-09-17 00:12:00
description: A list of all commonly asked questions I encountered in ML Interviews
tags: machine-learning
categories: machine-learning
tabs: true
nav: false
draft: true
---

# Technical questions (ML / DL / NLP / CV / LLM)

Below are categorized technical questions you might be asked in interviews. I grouped them so you can focus practice by area.

## Fundamentals & Math

1. Define bias–variance tradeoff and how you would diagnose it.
2. Derive gradient descent update for a simple linear regression and explain convergence conditions.
3. Explain cross-entropy loss and why it is used for classification.
4. What is softmax? Why can naive softmax be numerically unstable and how to fix it?
5. Explain L1 vs L2 regularization and when to use each.
6. What is Bayes’ theorem and how is it used in probabilistic models?
7. Show how backpropagation works for a 2-layer neural network (write equations).
8. Explain vanishing/exploding gradients and how batch norm / residual connections mitigate them.
9. What is the Jacobian and Hessian; why are they relevant to optimization?
10. Explain the difference between likelihood and posterior.

## Optimization & Training

11. Compare SGD, SGD+momentum, RMSProp, and Adam (advantages / failure modes).
12. What is learning rate scheduling? Describe step, cosine, linear warmup, and cyclical LR.
13. How does gradient clipping work and when is it used?
14. Explain batch normalization, layer normalization, and group normalization and where each is appropriate.
15. What is weight decay and how is it implemented in optimizers like AdamW?
16. How do you debug a training run where loss is NaN? Step-by-step checklist.
17. How do you choose batch size and how does it affect generalization and training stability?
18. What are common initialization schemes (Xavier/He) and why initialization matters.
19. Explain mixed-precision training and loss-scaling.
20. How to perform hyperparameter tuning at scale (Bayesian, grid, random)?

## Losses & Evaluation Metrics

21. Write the formulas for MSE, MAE, cross-entropy, focal loss, hinge loss.
22. When to use AUC vs accuracy vs precision/recall vs F1?
23. Explain mAP (mean Average Precision) and IoU in object detection.
24. What is NLL (negative log-likelihood) and how is it related to cross-entropy?
25. Explain BLEU, ROUGE, and METEOR for NLP evaluation — strengths/weaknesses.
26. How to evaluate a medical imaging model where false negatives are costly?
27. What is calibration and how to measure and improve it?
28. Explain confusion matrix and derived metrics.
29. How to evaluate ranked retrieval systems (MRR, NDCG, precision\@k).
30. How do you measure model robustness and adversarial vulnerability?

## Convolutional Neural Networks / Computer Vision

31. Explain what a convolution operation does (mathematically and intuitively).
32. What are depthwise separable convolutions and where are they used?
33. Describe residual blocks (ResNet) and why they enable deeper networks.
34. Explain dilated/atrous convolutions and use-cases.
35. Describe Feature Pyramid Networks (FPN) and why they help detection/segmentation.
36. Differences between FCN, U-Net, and UNet++ for segmentation.
37. How do object detectors like Faster R-CNN differ from SSD / YOLO?
38. How does Non-Maximum Suppression work? Problems and improvements.
39. Explain anchor boxes and anchor-free detection approaches.
40. What are common data augmentations for images and how can augmentation bias models?

## Transformer Architectures & Attention

41. Derive scaled dot-product attention and explain the role of the scaling factor.
42. What is multi-head attention and why multiple heads help?
43. Explain positional encodings and alternatives (learned, rotary).
44. What is layer normalization placement (pre-norm vs post-norm) and tradeoffs?
45. Describe transformer encoder vs decoder vs encoder-decoder.
46. How does attention complexity scale with sequence length and mitigation techniques (sparse, linear attention)?
47. Explain relative vs absolute positional encodings.
48. What is the mathematical form of softmax attention and its computational bottlenecks?
49. How do you implement causal attention for autoregressive models?
50. Explain FlashAttention (high-level) or any kernel-level speedups (if asked technically).

## NLP / Language Modeling / LLMs

51. What is perplexity and how is it computed? Pros and cons.
52. Explain autoregressive vs masked language models.
53. What is fine-tuning vs instruction-tuning vs parameter-efficient tuning (LoRA, adapters)?
54. Explain embeddings: how they are generated and evaluated.
55. What is retrieval-augmented generation (RAG) and pipeline components (retriever, index, reader)?
56. How to build a vector database and choose similarity metric (cosine vs dot vs L2)?
57. How to mitigate hallucinations in LLMs? List strategies.
58. What is chain-of-thought prompting and why does it help?
59. Explain concepts of few-shot and zero-shot learning with LLMs.
60. How to evaluate factuality of LLM outputs at scale?

## Self-supervised & Contrastive Learning

61. Explain contrastive loss (InfoNCE) and negative sampling.
62. What are SimCLR, MoCo, BYOL high-level differences?
63. How does contrastive learning apply to multimodal (image-text) setups?
64. What are pretext tasks for self-supervision in vision and NLP?
65. Describe masked autoencoding (MAE) and why it works.

## Generative Models (VAEs / GANs / Diffusion)

66. Explain the VAE objective and ELBO derivation.
67. What are mode collapse and instabilities in GANs? How to mitigate?
68. Explain diffusion models (forward + reverse process) at a high level.
69. Compare VAEs, GANs, and diffusion models for image generation.
70. How to evaluate generative models (FID, IS, human eval)?

## Sequence Models & Time Series

71. Compare RNN, LSTM, GRU, and Transformer for sequential data.
72. What are teacher forcing and scheduled sampling?
73. Explain temporal convolutional networks (TCN).
74. How to do time-series cross-validation (walk-forward validation)?
75. How to avoid leakage in time-series modeling?

## Graph ML

76. What is a GNN (GCN/GAT) and how message passing works?
77. How do you create graph features for fraud detection?
78. What is node2vec? How does it differ from classical embeddings?
79. Describe graph sampling strategies for scaling GNNs.
80. Explain transductive vs inductive graph learning.

## Model Compression, Serving & Inference

81. Describe quantization (post-training and quant-aware training).
82. What is pruning (structured vs unstructured) and how to retrain after pruning?
83. Explain knowledge distillation and student-teacher training.
84. How to design a low-latency inference pipeline for on-device LLMs?
85. How do you benchmark model latency and throughput? Important metrics.

## Safety, Privacy & Ethical ML

86. Define differential privacy (ε, sensitivity) and DP-SGD basics.
87. Explain federated learning high-level and challenges (heterogeneity, communication).
88. What is model inversion and membership inference? Mitigations?
89. How to audit a model for bias and unfairness? Procedure and metrics.
90. Considerations for medical/clinical ML deployment (regulatory, interpretability).

## Scaling, Distributed Training & Systems

91. What is data-parallel vs model-parallel training? Pros/cons.
92. Explain gradient accumulation and when to use it.
93. Describe pipeline parallelism and challenges.
94. How to handle large token contexts in training (memory & compute strategies).
95. Explain sharded optimizer states (ZeRO stages) at a high level.

## Practical Coding & Debugging / Whiteboard-style

96. Write pseudocode for a training loop with data loader, forward, loss, backward, optimizer step.
97. Given a model with sudden accuracy drop on validation, list systematic debug steps.
98. How to implement custom Dataset and DataLoader in PyTorch for multi-modal data?
99. How to profile a PyTorch training step and find bottlenecks?
100. Implement (or explain) a numerically-stable softmax cross-entropy in code.

## Advanced / Researchy

101. How to design an ablation study and report it convincingly?
102. Explain NTK or the lottery ticket hypothesis at a high level.
103. How to read and critique a ML paper (method, experiments, baselines)?
104. Tips to ensure reproducibility across hardware and randomness.
105. How would you formulate a novel research question in your domain (e.g., low-cost fundus screening)?

---

# Behavioral questions

(Use STAR: Situation, Task, Action, Result — prep concise 1–2 min answers)

1. Tell me about yourself / walk me through your resume.&#x20;
2. Describe a challenging technical problem you solved end-to-end.
3. Tell me about a time your project failed — what happened and what you learned?
4. Describe a conflict with a teammate and how you resolved it.
5. Give an example where you led or mentored others.
6. Describe how you prioritize work when you have multiple deadlines.
7. Tell me about a time you had to convince stakeholders to change direction.
8. Describe a time you improved a process (codebase, CI/CD, data pipeline).
9. How do you handle criticism of your code or design?
10. Tell me about a time you had to learn a new technology quickly for a project.
11. Describe a time when you made a tradeoff for product deadlines (quality vs speed).
12. Give an example of a time you took ownership beyond your job description.
13. How do you approach giving and receiving feedback?
14. Tell me about a challenging stakeholder (clinician, customer) in a healthcare project.
15. Describe a time you had to explain a complex technical idea to a non-technical audience.

---

# Targeted questions (project- & resume-specific)

Below are focused prompts tied to the projects and roles on your resume (use these to prepare deep dives & demos).&#x20;

## AIFred / Local coding assistant

1. Explain the full architecture (ASR → retriever → LLM → UI) and reasons for each component.
2. Why pick an on-device LLM (privacy/latency) and what compromises did you accept?
3. How did you select and build the retrieval corpus? Embedding model, chunking, indexing?
4. How did you measure and optimize end-to-end latency (numbers and techniques)?
5. How did you handle user privacy and local storage encryption?
6. Describe a bug/perf issue you solved in the Electron/Streamlit UI.
7. How would you extend the system to support multi-language code completion?
8. How do you keep the assistant up-to-date without uploading user code?

## FundusAI / Smartphone fundus imaging

9. Walk through the imaging pipeline — optics, capture, preprocessing, model inference.
10. How did you validate labels and what inter-rater agreement metrics did you use?
11. Describe the hardware adapter (3D-printed) constraints and how they affect model input.
12. What clinical metrics did you optimize for and why?
13. How did you ensure the model is robust to lighting and device variability?
14. What are deployment and regulatory considerations for a screening tool?
15. Present an example false positive & false negative and how you'd fix them.

## Document Tampering / YOLOv8 Project

16. Why YOLOv8 for tampering detection — performance vs complexity?
17. How did you annotate tampered regions and ensure label consistency?
18. How did you evaluate generalization to unseen tampering types?
19. How did you decide metric thresholds and tradeoff precision vs recall?
20. Describe data augmentation and synthetic tamper generation (if used).

## Fraud detection & Graph-RAG

21. Explain the graph schema you used (nodes, edges, attributes).
22. Walk through a production alert path: ingestion → scoring → human review.
23. How did you measure business impact (recovered \$ / false alert rate)?
24. How do you handle evolving fraud patterns and model drift?
25. Why use Graph-RAG for retrieval, and how does it improve answers over plain RAG?

## Open-source, Teaching & Writing

26. Examples of PRs you led — what was the technical challenge and community outcome?
27. How did you structure tutorials / workshops for non-research audiences?
28. How do you measure impact of technical writing (downloads, stars, adoption)?
29. How do you ensure documentation stays up to date with code?

## GPU / Performance / Triton (if on resume)

30. Explain kernel memory access patterns and common optimizations (tiling, shared memory).
31. Describe a profiling workflow (Nsight / profiler) and one optimization you implemented.
32. How would you convert a slow PyTorch op to a Triton kernel (high-level)?

---

If you want, I can now:

* Expand **any one** category into 50+ practice questions with answers.
* Generate concise **model answers / formulas** (math-focused) for the top 30 technical items.
* Produce **STAR-formatted behavioral answers** based on your resume content.

Which of those should I do next?
