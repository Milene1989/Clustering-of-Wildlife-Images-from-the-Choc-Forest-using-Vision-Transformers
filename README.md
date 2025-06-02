# 🐾 Clustering Wildlife Images from the Chocó Forest Using Vision Transformers and Unsupervised Learning

This repository contains the complete code and methodology for a thesis project focused on automatically grouping wildlife images captured by camera traps in the Chocó rainforest (Canandé Reserve, Ecuador). The project applies unsupervised learning techniques and state-of-the-art Vision Transformers to explore and organize large-scale image datasets without human annotations.

---

## 📌 Project Overview

This capstone project proposes a fully unsupervised pipeline to group images of wild animal species using Vision Transformers and clustering algorithms. Semantic embeddings are extracted using DINOv2 (ViT-B/14), and clustering is performed with K-Means and Gaussian Mixture Models (GMM). Final evaluations compare predicted clusters against real species labels (used for evaluation only) using metrics such as Adjusted Rand Index (ARI), Fowlkes-Mallows Index (FMI), and Jaccard Index.

The project aims to support biodiversity research by reducing dependence on manual labeling, facilitating faster exploration of ecological camera trap datasets.

---

## 🧠 Technologies Used

- Python 3.10  
- [DINOv2 Vision Transformers](https://github.com/facebookresearch/dinov2) (ViT-B/14, Meta AI)  
- PyTorch  
- scikit-learn (clustering, evaluation metrics)  
- Matplotlib & Seaborn (visualizations)  
- t-SNE (for dimensionality reduction and visual inspection)  
- Data provided by Fundación Jocotoco, previously processed using MegaDetector  

---

## 📂 Dataset Description

- 40,000+ cropped wildlife images obtained from camera trap videos
- Bounding boxes generated using [MegaDetector](https://github.com/microsoft/CameraTraps/blob/main/megadetector.md)
- Images categorized into:
  - `train/`
  - `validation/`
  - `test/`
- Ground truth species labels were used **only for evaluation**, not during training
- Species include:
  - Central American agouti (*Dasyprocta punctata*)
  - Nine-banded armadillo (*Dasypus novemcinctus*)
  - Lowland paca (*Cuniculus paca*)
  - Squirrels and small rodents
  - Great tinamou (*Tinamus major*)

---

## ⚙️ Pipeline Overview

```text
1. Preprocessing
   └── Cropping image regions based on bounding boxes (from MegaDetector)

2. Feature Extraction
   └── Embedding each image using DINOv2 (ViT-B/14, 768-dim vector)

3. Clustering
   └── Applying K-Means and GMM with multiple values of K
   └── Model selection via metrics (Silhouette, AIC, BIC)

4. Evaluation
   └── External metrics: ARI, FMI, Jaccard
   └── Visualizations: t-SNE plots, confusion matrices

5. Interpretation
   └── Qualitative inspection of cluster contents
   └── Taxonomic alignment and error analysis


---

## 📈 Clustering Evaluation Metrics

| Metric                  | K-Means (k=6) | GMM (k=3) |
|-------------------------|---------------|-----------|
| Adjusted Rand Index (ARI) | **0.7198**   | 0.7309    |
| Jaccard Index (macro)     | **0.5108**   | 0.3692    |
| Fowlkes-Mallows Index (FMI) | 0.7930     | **0.8222** |

Although GMM shows slightly higher performance in ARI and FMI, K-Means significantly outperforms in Jaccard Index, indicating better alignment in cluster composition. Additionally, K-Means provides finer-grained segmentation across species, especially for visually similar ones.

---

## 📊 Sample Visualizations

- **t-SNE plots** with real species as color and cluster ID as overlay
- **Confusion matrices** comparing true species vs. predicted cluster assignments
- **Cluster-wise image grids** illustrating visual homogeneity or misclassifications

You can find these plots in the final thesis document.

---

## 📝 Citation

If you use this repository, model pipeline, or visualizations in academic or applied work, please acknowledge this project. The underlying camera trap data was provided by Fundación Jocotoco and processed using MegaDetector as part of a previous classification study by Edwin Montenegro.

---

## 👩‍�

