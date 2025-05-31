<img src="assets/banner.png" alt="Wildlife Monitoring System Banner" style="width:100%;"/>

# Clustering of Wildlife Images from the Chocó Forest using Vision Transformers and Unsupervised Learning Techniques

A complete unsupervised machine learning pipeline for organizing and exploring large-scale camera trap imagery from the Chocó Forest (Canandé Reserve) using Vision Transformers and clustering algorithms.

---

## 📌 Project Overview

This capstone project implements a fully unsupervised system for automatically grouping images of wild species captured by camera traps in the Ecuadorian Chocó. We use **DINOv2 (ViT)** models to extract semantic features, followed by **K-Means** and **Gaussian Mixture Models (GMM)** for clustering. The clusters are then evaluated against known species labels using metrics such as ARI, FMI, and Jaccard Index.

---

## 🧠 Technologies Used

- **Python 3.8+**
- **PyTorch**
- **DINOv2 Vision Transformers (Meta AI)**
- **Scikit-learn** for clustering & evaluation
- **t-SNE** for dimensionality reduction and visualization
- **Camera trap data** provided by the **Jocotoco Foundation**

---

## 📂 Dataset Description

- 40,000+ cropped animal images from trap videos
- Bounding boxes extracted using MegaDetector
- Data split into:  
  - `train/`  
  - `validation/`  
  - `test/`  
- Ground truth species labels used only for evaluation


## ⚙️ Pipeline Overview

1. Preprocess and crop images from bounding boxes  
2. Extract 768-dim feature vectors using DINOv2  
3. Apply KMeans and GMM clustering  
4. Evaluate clustering performance  
5. Visualize clusters using t-SNE and confusion matrices  

📊 Full architecture diagram:  
![Architecture](assets/system_architecture.png)

---

## 📈 Evaluation Metrics

| Metric                   | K-Means | GMM |
