Based on the directory structure you provided, I have integrated the `Project Structure` section and embedded the corresponding images into the `Results` section.

Here is the complete, updated `README.md` file ready for your GitHub repository.

```markdown
# MAGE-V-Pro: An Open and Reproducible Multimodal AI Framework for Skin Disease Diagnosis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Accepted_%2F_In_Press-success)]()
[![Journal](https://img.shields.io/badge/Journal-Procedia_Computer_Science-blue)](https://www.sciencedirect.com/journal/procedia-computer-science)

## 📋 Abstract
[cite_start]This repository contains the official implementation of **MAGE-V-Pro** (Multimodal Attention-Guided Encoder for Vision)[cite: 13, 14]. [cite_start]Accurate diagnosis of dermatological conditions is challenging due to high inter-class visual similarity and intra-class variability[cite: 11]. [cite_start]While deep learning has progressed, unimodal (vision-only) systems often fail to capture contextual cues embedded in medical narratives[cite: 12].

[cite_start]MAGE-V-Pro integrates vision features (**DINOv2**) with biomedical text embeddings (**PubMedBERT**) using Feature-wise Linear Modulation (**FiLM**)[cite: 14]. [cite_start]The model achieves **95.71% test accuracy**, demonstrating a 24.28 percentage point improvement over vision-only baselines[cite: 17].

## 📂 Project Structure
The repository is organized as follows:

```text
MAGE-V-Pro/
│
├── mage-v-final-research_2.0.ipynb    # 🧠 Main research notebook (Full Pipeline)
├── mage-v-final-research_2.0.ipynb - pdf.pdf  # 📄 PDF export of the notebook
├── Presenter Certificate - FTNCT'08.pdf       # 🏆 Conference Presentation Proof
├── README.md                          # 📖 Project Documentation
│
├── Confirmed_results/                 # 📊 Generated plots and metrics
│   ├── stage2_test_confusion_matrix.png
│   ├── stage-2_test_eval_metrics.png
│   ├── training_curves_S2.png
│   ├── attention_map_visualizations/  # 👁️ FiLM Attention Maps
│   └── grad_cam_visualizations/       # 🔍 Grad-CAM Explainability
│
├── Gemini_generated_text_descriptions/
│   └── text_final.json                # 📝 Synthetic patient descriptions (Source)
│
└── RAG/
    └── kb_new.json                    # 📚 Knowledge Base for fallback logic

```

## 🎯 Problem Statement

Dermatology presents unique diagnostic challenges:

* 
**Inter-class similarity:** Visually similar lesions may represent different diseases.


* 
**Intra-class variability:** The same disease can appear differently across patients.


* 
**Context dependency:** Clinical diagnosis relies on both visual examination and patient history.



Purely vision-based models ignore the rich semantic information clinicians use. MAGE-V-Pro addresses this by processing visual and textual data simultaneously to mimic expert diagnostic reasoning.

## 📊 Dataset

This study utilizes publicly available datasets to ensure reproducibility.

* 
**Source:** [DermNet Dataset](https://www.kaggle.com/datasets/shubhamgoel27/dermnet) 


* 
**Classes (7):** Acne, Psoriasis, Eczema, STDs, Fungal Infections, Basal Cell Carcinoma (BCC), Seborrheic Keratosis.


* 
**Text Modality:** Patient-style clinical descriptions generated via Gemini LLM (validated for clinical accuracy) to simulate real-world narratives without privacy risks.



| Split | Size | Purpose |
| --- | --- | --- |
| **Stage 1** | 5,605 images | Vision encoder domain adaptation 

 |
| **Stage 2** | 2,030 pairs | Multimodal fine-tuning 

 |
| **Test Set** | 70 pairs | Final evaluation (held-out) 

 |

## 🏗️ Methodology

MAGE-V-Pro employs a two-stage training strategy combining three core modules:

### 1. Architecture Components

* 
**Vision Encoder:** **DINOv2 (ViT-S/14)** self-supervised Vision Transformer (384-dim embeddings).


* 
**Text Encoder:** **PubMedBERT** optimized using **LoRA** (Low-Rank Adaptation) for parameter efficiency.


* 
**Fusion Module (FiLM):** Conditions visual features on textual context using:





### 2. Training Strategy

* 
**Stage 1 (Vision-Domain Adaptation):** Fine-tuning DINOv2 on dermatology images (20 epochs, weighted cross-entropy).


* 
**Stage 2 (Multimodal Fine-Tuning):** Integrating vision and text encoders with FiLM fusion (30 epochs).


* 
**RAG Fallback:** A **Confidence-Based Retrieval-Augmented Generation** mechanism activates when prediction confidence is < 0.6.



## 📈 Results

The multimodal approach significantly outperforms the vision-only baseline.

### Quantitative Performance

| Model | Accuracy | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- |
| **Vision-Only Baseline** | 71.43% | 76.18% | 71.43% | 71.34%

 |
| **MAGE-V-Pro (Ours)** | **95.71%** | **95.97%** | **95.71%** | <br>**95.62%** |

**Key Findings:**

* 
**Resolves Ambiguity:** Effectively distinguishes morphologically similar conditions (e.g., BCC vs. Seborrheic Keratosis).


* 
**Improved Localization:** Grad-CAM visualizations show more focused attention on diagnostically relevant lesion features.



#### Confusion Matrix Comparison

The multimodal model (Stage 2) resolves ambiguity between morphologically similar classes.

| Stage 1 (Vision-Only) | Stage 2 (Multimodal - Ours) |
| --- | --- |
|  |  |

### Qualitative Analysis (Explainability)

#### 1. FiLM Attention Maps

The model learns to weigh specific visual features based on the text description.

#### 2. Grad-CAM Visualizations

Heatmaps show the model focusing on the lesion area.

## 🚀 Reproducibility

This project is designed for full reproducibility using public resources.

### Prerequisites

* 
**GPU:** NVIDIA Tesla T4 (or equivalent) 


* **Python:** >= 3.8
* 
**Key Libraries:** `torch`, `transformers`, `peft`, `torchvision`, `scikit-learn`.



### Installation

```bash
# Clone the repository
git clone [https://github.com/yourusername/MAGE-V-Pro.git](https://github.com/yourusername/MAGE-V-Pro.git)
cd MAGE-V-Pro

# Install dependencies
pip install -r requirements.txt

```

### Quick Start

To run the complete training and evaluation pipeline:

```bash
jupyter notebook mage-v-final-research_2.0.ipynb

```

## 📧 Contact

For questions or collaboration inquiries:

* Chola Chetan Chukkala  (Author): vpscholachetan24@gmail.com
---

Note: This paper is accepted for publication in **Procedia Computer Science** (Elsevier) at the **Eighth International Conference on Future Trends in Networking and Computing Technologies (FTNCT-08)**. See `Presenter Certificate - FTNCT'08.pdf` for verification.

```

```