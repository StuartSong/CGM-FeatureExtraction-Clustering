# CGM-FeatureExtraction-Clustering
Code for analyzing Continuous Glucose Monitoring (CGM) data using day segmentation, feature extraction, and clustering methods like UMAP. This project focuses on identifying glycemic patterns to support personalized diabetes management for diabete patients

Overview

The study focuses on applying Uniform Manifold Approximation and Projection (UMAP) to CGM data, allowing for clustering analysis that distinguishes between T1D individuals on insulin and healthy controls (HC). By extracting glycemic features and applying dimensionality reduction, we explore the differences in glycemic profiles and propose a novel approach to personalized diabetes care.

Key highlights of this project include:

Clustering glycemic profiles using UMAP.

Quantifying differences between T1D and HC using Silhouette Scores.

Providing an easily interpretable metric for evaluating diabetes management.

Graphical Abstract

(Include graphical abstract here)

How to Cite

If you use any part of this repository in your research, please cite our manuscript:

J. Song, J. McNeany, Y. Wang, T. Daley, A. Stecenko, R. Kamaleswaran, "Riemannian Manifold-based Geometric Clustering of Continuous Glucose Monitoring to Improve Personalized Diabetes Management," Computers in Biology and Medicine, vol. 183, 2024, pp. 109255. doi:10.1016/j.compbiomed.2024.109255

Installation

To get started, clone the repository and install the required dependencies:

Usage

This repository includes the following scripts:

CGM_TAML_PA180_2.py: Python script for data preprocessing and temporal segmentation of CGM data. It includes functions for cleaning CGM datasets, handling missing values, and dividing the data into daily segments to facilitate further analysis.

cgmquantify_stuart.py: Script for feature extraction and analysis of glycemic variability.

Sample Training and Validation Dataset Creation.ipynb: Jupyter notebook demonstrating the creation of training and validation datasets, as well as UMAP-based clustering.

Results

The results indicate that UMAP effectively distinguishes between glycemic profiles of individuals with T1D and healthy controls. The clustering results suggest potential use for personalized care by understanding and interpreting day-by-day glycemic control.

License

This project is licensed under the MIT License - see the LICENSE file for details.

