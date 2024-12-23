# **Pneumonia Detection Using Chest X-ray Images**

## Overview

This repository provides a solution for detecting pneumonia using chest X-ray scans. It uses a Convolutional Neural Network (CNN) model trained on labeled chest X-ray images to classify scans as either PNEUMONIA or NORMAL. The project is aimed at aiding medical professionals in diagnosing pneumonia more effectively and efficiently.

## Project Description

The Pneumonia Detection system leverages deep learning techniques to analyze chest X-ray images and determine the presence of pneumonia. The project includes the following features:

- **Binary Classification**: The model distinguishes between X-ray scans of healthy lungs and those showing signs of pneumonia.

- **Data Augmentation**: Training data is augmented to improve model robustness and generalization.

- **Custom Pretrained Model**: A pretrained model has been fine-tuned for this specific task to achieve higher accuracy.

This system can be used as a diagnostic aid, particularly in areas with limited access to experienced radiologists.

## Use Cases

1. Medical Diagnosis Assistance: Assists healthcare providers in identifying pneumonia from chest X-rays.

2. Screening in Remote Areas: Facilitates preliminary screenings in regions with limited medical infrastructure.

3. Training and Education: Provides a tool for medical students and professionals to learn about pneumonia detection using imaging.

## Technical Details

Model Architecture

The project employs a custom CNN architecture fine-tuned for binary classification of chest X-ray images. It is built using TensorFlow/Keras and incorporates the following layers:

- **Convolutional Layers**: Extract features from the input images.

- **Pooling Layers**: Reduce spatial dimensions for computational efficiency.

- **Fully Connected Layers**: Map extracted features to classification labels.

## Dataset

The dataset includes chest X-ray images categorized into two classes:

- **NORMAL**: X-ray images of healthy lungs.

- **PNEUMONIA**: X-ray images showing signs of pneumonia.

The dataset is preprocessed to normalize image sizes and enhance image quality. Data augmentation techniques such as rotation, flipping, and zooming are applied to improve model generalization.

## Training Details

1. Data Preprocessing:

- Resizing images to a fixed resolution.

- Normalizing pixel values to [0, 1].

2. Model Training:

- Loss Function: Binary Cross-Entropy

- Optimizer: Adam

- Learning Rate: 1e-4

- Batch Size: 32

- Epochs: 50

3. Evaluation Metrics:

- Accuracy

- Precision, Recall, and F1-Score
