# **Riemannian Manifold-based Geometric Clustering of Continuous Glucose Monitoring Data**

This repository contains the code and data for the study titled **"Riemannian Manifold-based Geometric Clustering of Continuous Glucose Monitoring to Improve Personalized Diabetes Management"**. The project applies Riemannian manifold-based geometric clustering, to better understand glycemic control in individuals with Type 1 Diabetes (T1D) and healthy controls using Continuous Glucose Monitoring (CGM) data. The approach aims to enhance personalized diabetes management. The repository includes code for daily segmentation, glycemic feature extraction, UMAP projection, and silhouette score calculation.

![Graphical Abstract](Graphical_Abstract.jpg)

---

### **Key Highlights**
- **Glycemic Feature Extraction**: Includes an easy-to-use function to compute 29 key glycemic features from CGM data.
- **UMAP Clustering**: Utilizes UMAP for effective visualization and separation of glycemic profiles.
- **Silhouette Score Analysis**: Quantifies the differences between T1D and HC clusters using Silhouette Scores.
- **Personalized Care**: Provides interpretable metrics to support more personalized diabetes management.

---

## **Installation**

Follow these steps to get started:

1. Clone the repository:
    ```bash
    $ git clone https://github.com/username/cgm_clustering.git
    $ cd cgm_clustering
    ```

2. Install the required dependencies:
    ```bash
    $ pip install -r requirements.txt
    ```

---

## **Usage**

The repository includes the following scripts for processing and analyzing CGM data:

- **`CGM_TAML_PA180_2.py`**: Handles data preprocessing, cleaning, and temporal segmentation of CGM data.
- **`cgmquantify_stuart.py`**: Extracts glycemic variability features and computes metrics for analysis.
- **`Sample Training and Validation Dataset Creation.ipynb`**: Demonstrates the creation of datasets for training and validation, alongside UMAP-based clustering.

### **To Run the Code**
Simply execute any of the Python scripts or open the Jupyter notebook in your favorite environment for experimentation.

---

## **How to Cite**

If you find this repository helpful in your research, please cite the manuscript as follows:

> **Song J., McNeany J., Wang Y., Daley T., Stecenko A., Kamaleswaran R.**, *"Riemannian Manifold-based Geometric Clustering of Continuous Glucose Monitoring to Improve Personalized Diabetes Management,"* **Computers in Biology and Medicine**, vol. 183, 2024, pp. 109255.  
> [doi:10.1016/j.compbiomed.2024.109255](https://doi.org/10.1016/j.compbiomed.2024.109255)

---

## **Contributing**

We welcome contributions! Please submit a pull request if you'd like to improve the code, add features, or fix bugs.

---

### **Contact**

For questions, suggestions, or feedback, feel free to reach out to Jiafeng via email: [sjfsjf2010@gmail.com](mailto:sjfsjf2010@gmail.com).
"""

