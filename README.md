# Temporal Action Segmentation with TCNs (PyTorch)

An PyTorch implementation of Temporal Convolutional Networks (TCNs) for dense frame-wise **Temporal Action Segmentation** in untrimmed videos. 

This repository explores several core architectural variants based on the state-of-the-art **MS-TCN (Multi-Stage Temporal Convolutional Network, CVPR 2019)**, analyzing the impacts of receptive fields, cross-stage feature fusion, global video-level regularizations, and multi-scale temporal modeling.

---

## Key Features

- **Single-Stage TCN**: Deep 1D dilated residual convolutional neural network with linear dilation factor progression.
- **Multi-Stage Refinement**: 4-stage sequential refinement with probability-to-feature concatenation to balance local semantic features and transition contexts.
- **Video-Level Auxiliary Loss**: Multi-label global regularizer implementing temporal max-pooling and Binary Cross-Entropy (BCE) loss to eliminate out-of-context segments.
- **Parallel Multi-Scale TCN**: Multi-branch resolution architecture with 1x, 4x, and 8x downsampled temporal spaces, utilizing high-level feature upsampling and average fusion.

---

## Requirements and Installation

- Python 3.x
- PyTorch >= 1.1
- NumPy

Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/Temporal-Action-Segmentation-TCN-PyTorch.git
cd Temporal-Action-Segmentation-TCN-PyTorch
```

---

## Repository Structure

```text
├── dataset.py          # Custom PyTorch Dataset with temporal sequence padding (collate_fn)
├── model.py            # Complete implementations of Single-Stage, Multi-Stage, and Multi-Scale TCNs
├── main.py             # Unified entry point for training and predicting across different tasks
├── eval.py             # Script to evaluate model performance on test set
├── metrics.py          # Metric calculations (Mean of Frames, Segmental Edit Distance, Segmental F1@10,25,50)
└── README.md           # Documentation
```

*Note: The actual dataset folder (e.g. `data/`) contains heavy feature directories and is excluded from version control.*

---

## Performance Benchmarks (Quantitative Results)

The models were evaluated on the test set of dense action segmentation sequences. The table below outlines the quantitative results for all implemented architectures:

| Configuration | MoF (Acc) | Segmental Edit | F1@10 | F1@25 | F1@50 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Q1: Single-Stage TCN (Linear Dilation)** | 50.14% | 12.42% | 7.97% | 6.36% | 4.03% |
| **Q2: Multi-Stage TCN (Feature Concat)** | 54.60% | 16.46% | 11.71% | 9.79% | 6.58% |
| **Q3: Q2 + Video-Level BCE Loss** | 53.15% | 15.71% | 10.11% | 8.56% | 5.93% |
| **Q4: Parallel Multi-Scale TCN** | **69.65%** | **35.54%** | **26.60%** | **23.73%** | **18.39%** |

---

## Key Academic Insights

1. **The Receptive Field Bottleneck**: 
   Under a linear dilation factor ($d_l = l$, $L=10$), the receptive field is physically limited to only **111 frames** (~7.4 seconds at 15 fps), leading to heavy **over-segmentation** (very low Edit & F1 scores) as the model lacks long-term context.
   
2. **The Power of Multi-Scale modeling**:
   By downsampling the temporal inputs by factors of 4 and 8, the multi-scale architecture (Q4) effectively increases the branch receptive fields to **444 and 888 frames** respectively, without increasing parameter counts. This scale-specific context learning, combined with high-level feature alignment, more than **doubles** the segmental Edit distance and F1 scores.

---

## How to Run

### 1. Data Preparation
Place your pre-extracted feature directory and annotations in the `data/` folder following the hierarchy below:
```text
data/
├── features/          # numpy arrays (.npy) with dimension [2048 x T]
├── groundTruth/       # ground truth frame-wise text labels (.txt)
├── mapping.txt        # class name to ID mapping
├── train.bundle       # training video names
└── test.bundle        # testing video names
```

### 2. Training and Prediction
You can train and generate predictions for any task using the unified command line interface in `main.py`:

#### Question 1 (Single-Stage TCN)
```bash
python main.py --task q1 --action train
python main.py --task q1 --action predict
python eval.py --pred_path ./predictions_q1
```

#### Question 2 (Multi-Stage TCN with Concatenation)
```bash
python main.py --task q2 --action train
python main.py --task q2 --action predict
python eval.py --pred_path ./predictions_q2
```

#### Question 3 (Multi-Stage with Video-Level Loss)
```bash
python main.py --task q3 --action train
python main.py --task q3 --action predict
python eval.py --pred_path ./predictions_q3
```

#### Question 4 (Multi-Scale TCN)
```bash
python main.py --task q4 --action train
python main.py --task q4 --action predict
python eval.py --pred_path ./predictions_q4
```

---


## References
1. **Multi-Stage Temporal Convolutional Network (MS-TCN)**:
   * Yazan Abu Farha and Juergen Gall. *"MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation."* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.
   * [ArXiv Paper Link](https://arxiv.org/abs/1903.01945)

---

## Author
*   **Jinglin Zhu** - [GitHub Profile](https://github.com/jinglin-zhu) - *Implementation, Evaluation, and Documentation*
*   This project was developed as part of the **Video Analytics (SS26)** course at the **University of Tübingen**.

---

## Disclaimer
This repository is an independent PyTorch re-implementation of the MS-TCN and Multi-Scale TCN architectures, developed solely for academic and educational purposes. All rights, patents, and intellectual property regarding the original algorithms, network designs, and methodologies belong to the respective authors of the landmark papers cited in the references section above.
