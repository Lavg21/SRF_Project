# 🚗 Vehicle Classification Project

> A machine learning-based approach for vehicle classification using k-NN and color histograms.

## 📌 Overview

This project focuses on **classifying vehicles** from images using **bounding-box 2D** and a **k-NN classification model** based on **color histograms**. The system aims to categorize vehicles into predefined classes such as **sedan, minivan, and bus**.

## 🛠 Features

✅ **Vehicle detection and classification**  
✅ **k-NN classifier based on color histograms**  
✅ **RGB and HSV histogram analysis**  
✅ **Dataset processing and augmentation**  
✅ **Evaluation of model accuracy**  

## 📜 Table of Contents

- [Dataset](#dataset)
- [Methodology](#methodology)
- [Experiments](#experiments)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)

## 📂 Dataset

The dataset consists of vehicle images obtained from [Roboflow Universe](https://universe.roboflow.com/ana-lowela-l--lucas/vehicle-classification-sgcum). The dataset is formatted in JSON, with annotations specifying vehicle categories. 

### Dataset Preprocessing:
- **Data Extraction:** Images are categorized based on vehicle type.
- **File Renaming:** Images are renamed systematically for consistency.
- **Folder Structuring:** Data is divided into training and testing sets.

## 🏗 Methodology

### 1️⃣ Color Histogram Calculation
- Extracts color distribution from images in **RGB** and **HSV** color spaces.
- Normalizes histograms for consistent feature representation.

### 2️⃣ k-NN Classifier
- Computes distance between test images and dataset using histogram comparison.
- Uses **majority voting** to assign the most frequent class among k-nearest neighbors.

### 3️⃣ Classification and Evaluation
- Compares classification performance using **accuracy metrics**.
- Evaluates k-NN's efficiency in real-world classification tasks.

## 🔬 Experiments

- **RGB vs. HSV Comparison:** Evaluates histogram effectiveness in different color spaces.
- **Varying k-values:** Tests accuracy sensitivity with different neighbor counts.
- **Dataset Variations:** Analyzes classification performance under different lighting conditions.

## 📊 Results

- The use of **HSV histograms** showed improved classification accuracy.
- The optimal **k-value** for classification was determined experimentally.
- Dataset balance and augmentation improved overall classifier robustness.

## 💻 Installation

### Prerequisites
- Python 3.8+
- OpenCV
- NumPy
- Matplotlib

## ⚡ Technologies Used

- **Machine Learning:** k-NN Classifier
- **Image Processing:** OpenCV
- **Data Handling:** JSON, NumPy
- **Visualization:** Matplotlib

## 🔮 Future Improvements

🔹 Optimize k-NN performance with advanced distance metrics.  
🔹 Experiment with **CNN-based** classification for improved accuracy.  
🔹 Expand dataset to include more vehicle categories.
