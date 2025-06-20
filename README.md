# Explainable Saliency: Articulating Reasoning with Contextual Prioritization

This repository contains the code and resources for the CVPR 2025 paper: **Explainable Saliency: Articulating Reasoning with Contextual Prioritization**. Our work introduces a novel saliency prediction model that not only identifies important image regions but also explains its reasoning process using a vision-language approach and contextual prioritization.
<!-- 
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://example.com/paper)  
[![Code](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/your-repo-link)  
 -->

---

## 🔍 **Overview**

Our approach bridges the gap between saliency prediction and interpretability by incorporating explicit reasoning and contextual prioritization mechanisms:

- **Explicit Reasoning**: Uses a vision-language model to generate semantic proposals and natural language explanations that mirror human attention and problem-solving processes.
- **Contextual Prioritization**: Dynamically selects the most relevant features to focus on key semantic elements in the image.

### Key Features:
1. **Saliency Prediction with Reasoning**: Produces state-of-the-art saliency maps alongside explanations for why specific regions are prioritized.
2. **Human-Like Interpretability**: Generates natural language descriptions that explain the model’s decisions in a manner aligned with human reasoning.
3. **Dynamic Contextual Focus**: Adapts to scene complexity, emphasizing only the most relevant semantic elements.



## ⚙️ **Setup**

### Prerequisites:
- Python 3.8
- PyTorch 1.9.1
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## 📂 *Dataset Preparation:*
The `air_data` dataset can be downloaded from the following link:  
[Download air_data](https://drive.google.com/file/d/1Ex_eQv79B4HJe3eTudZ9XtoEySbJDU-E/view?usp=sharing)
Structure the root directory as follows:
```
Explainable_Saliency/
  |-- data/
      |-- air_data/
      |-- osie_data/
  |-- src/
  |-- bash/
  |-- README.md
  |-- requirements.txt
```

<!-- Preprocess the data:
```bash
python preprocess_dataset.py --dataset <dataset_name>
``` -->

---

## 🚀 **Training**

Train the model on both datasets:
```bash
python ./src/AirReasonTrainer.py --mode=train --epoch=10 --lr=4e-4 --batch_size=10 --topk=3 --checkpoint_dir=./workdir
```


## 📊 **Evaluation**

Evaluate the trained model:
```bash
python AirReasonTrainer.py --mode=eval --topk=3 --checkpoint_dir=./workdir/your_checkpoints_path
```

---

## ✍️ **Citation**

If you find our work helpful in your research, please consider citing it:
```bibtex
@InProceedings{Chen_2025_CVPR,
    author    = {Chen, Nuo and Jiang, Ming and Zhao, Qi},
    title     = {Explainable Saliency: Articulating Reasoning with Contextual Prioritization},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {9601-9610}
}
