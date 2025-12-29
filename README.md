#  An Ultra-Lightweight Cross-scale Attention Mamba Network for Accurate Skin Lesion Segmentation

This is the official implementation of **UCA-MNet**, a Cross-scale Attention Mamba Network as introduced in the paper:

**"An Ultra-Lightweight Cross-scale Attention Mamba Network for Accurate Skin Lesion Segmentation"**.

📄 **Paper:** [IEEE Journal of Biomedical and Health Informatics Link to be added] 

🚀 **Key Highlights:** 0.33M Parameters | 4.3 GFLOPs | 0.0537s Inference Time 

UCA-MNet addresses the computational-precision trade-off by utilizing **Bi-Mamba** to model long-range spatial dependencies with linear complexity. It is approximately **83 times more parameter-efficient** than leading models like VM-UNet while maintaining clinical-grade precision.

---

## 🏗️ Architecture

UCA-MNet achieves a state-of-the-art balance between accuracy and efficiency through three key innovations:

1. **Multi-Scale Module (MSM):** The core computational engine utilizing bidirectional Mamba (Bi-Mamba) to efficiently model long-range spatial dependencies.
2. **Hierarchical Feature Encoder:** A sequence of four Encoder Blocks (E-Blocks) that combine query/key projections with expansion convolutions to extract rich multi-scale features.
3. **Feature Compression and Fusion Module (FCFM):** Aggregates and synthesizes outputs from all decoder scales into a single, coherent high-resolution segmentation map.





---

## 📊 Performance Results

UCA-MNet delivers superior or comparable performance to leading heavyweight architectures with a fraction of the parameters.

### Quantitative Comparison

| Dataset | F1-Score | mIoU | Parameters | GFLOPs |
| --- | --- | --- | --- | --- |
| ISIC-2017 | 0.8254 | 0.8180 | 0.33M | 4.30 |
| ISIC-2018 | 0.8814 | 0.8515 | 0.33M | 4.30 |
| PH2 | 0.9202 | 0.8604 | 0.33M | 4.30 |

Data compiled from Tables III, IV, and V of the manuscript.

### Qualitative Results

UCA-MNet produces more precise lesion boundaries and sharper edge delineation, effectively capturing structural details across diverse imaging conditions.

---

## 🚀 Quick Start

### 1. Requirements

* Python: 3.7.5+ 


* PyTorch: 2.2.0 


* NVIDIA GPU with 8GB VRAM (e.g., GTX 1070) 


* NumPy, SciPy, Matplotlib, OpenCV 



### 2. Experimental Setup

* 
**Preprocessing:** Images and masks are standardized to  pixels.


* 
**Optimizer:** Adam with an initial learning rate of .


* 
**Loss Function:** Hybrid loss combining Binary Cross-Entropy (BCE), Dice, and IoU losses.



---

## 📁 Dataset Preparation

We utilize three publicly available benchmarks for skin lesion segmentation:

### Supported Datasets

1. **ISIC-2017 Segmentation Dataset**
* 2,000 dermoscopic images.


* [Download from ISIC Archive](https://challenge.isic-archive.com/data/)


2. **ISIC-2018 Task 1: Lesion Boundary Segmentation**
* 2,594 training images.


* [Download from ISIC Archive](https://challenge.isic-archive.com/data/)


3. **PH2 Dataset**
* 200 images from Pedro Hispano Hospital.


* [Download Link](https://www.fc.up.pt/addi/ph2%20database.html)



### Directory Structure

```
UCA-MNet/
├── data/
│   ├── ISIC2017/
│   │   ├── images/
│   │   └── masks/
│   ├── ISIC2018/
│   │   ├── images/
│   │   └── masks/
│   └── PH2/
│       ├── images/
│       └── masks/
├── models/
├── utils/
└── train.py

```

---

## 🎯 Training

To train UCA-MNet on a specific dataset using an 80-10-10 split:

```bash
# Train on ISIC-2017
python train.py --dataset ISIC2017 --epochs 100 --batch_size 2

# Train on ISIC-2018
python train.py --dataset ISIC2018 --epochs 100 --batch_size 2

# Train on PH2
python train.py --dataset PH2 --epochs 100 --batch_size 2

```

---

## 🧪 Evaluation

To evaluate a trained model using AIU (Average Intersection Over Union) and OIS (Optimal Image Score) metrics:

```bash
python evaluate.py --dataset ISIC2018 --model_path checkpoints/best_model.pth

```

---

## 📝 Citation

If you find this work useful for your research, please cite our manuscript:

```bibtex
@article{alharith2025ucamnet,
  title={An Ultra-Lightweight Cross-scale Attention Mamba Network for Accurate Skin Lesion Segmentation},
  author={Alharith, Razan and Zhang, Jiashu and Zhao, Chengqiang},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2025}
}

```

---

## ✉️ Contact

**Razan Alharith** (Southwest Jiaotong University)

Email: [razanalharith@my.swjtu.edu.cn](mailto:razanalharith@my.swjtu.edu.cn)
