## Gaze-Informed Dual-View Preprocessing for Efficient Egocentric VQA

This repository contains the code for our **CS 381V (Graduate Visual Recognition) final project** on gaze-informed dual-view preprocessing for efficient egocentric video question answering (VQA).

We propose a dual-stream visual representation for pretrained Vision-Language Models (VLMs) such as **Qwen3-4B-VL**:

1. A **downsampled full-frame view** that preserves global context.
2. A **gaze-centered crop** that captures high-resolution detail near the fixation point.

Both views are resized to the same resolution, interleaved in time, and processed by the vision tower. This reduces the number of vision tokens compared to using high-resolution full frames while retaining important scene information.

---

## Method Overview

- Backbone: **Qwen3-4B-VL**
- Trainable components: **vision projector (~20M parameters)**
- Dataset: **HD-EPIC** egocentric VQA
- Training: projector-only finetuning for 1 epoch on 2,447 QA pairs
---

## Zero-Shot Results on HD-EPIC (Qwen3-4B-VL)

| Method                                         | Vision Tokens | HD-EPIC Accuracy (%) | Inference Time (s) |
|-----------------------------------------------|---------------|----------------------|--------------------|
| Random Guessing                               | --            | 20.00                | --                 |
| (4×256×256) Downsampled Frames                | 512           | 23.94                | 2.88               |
| (4×256×256) Gaze-Cropped Frames               | 512           | 21.81                | 2.87               |
| (4×512×512) Downsampled Frames                | 2048          | 23.94                | 10.46              |
| **(8×256×256) Dual-View (Ours)**              | **1024**      | **24.47**            | **5.39**           |
| (4×512×512) Gaze-Cropped Frames               | 2048          | **30.32**            | 10.47              |

Our dual-view configuration offers better accuracy than all single-view 256 baselines and competitive performance relative to 512 baselines, while using half as many vision tokens as 512×512 inputs and lower inference time.

---

## Finetuning Summary

After one epoch of projector-only finetuning:

| Method                 | Vision Tokens | Accuracy (%) |
|------------------------|---------------|--------------|
| Original 512×512       | 2048          | 28.17        |
| **Dual-View 256×256**  | **1024**      | **28.78**    |

The dual-view setting remains slightly more accurate while using half the visual token budget.

