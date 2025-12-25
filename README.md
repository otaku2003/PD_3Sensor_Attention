Attention-Based Multi-Sensor CNN for PD Localization
This repository provides the complete pipeline to reproduce the results of an

attention-based three-sensor CNN model for partial discharge localization.

Repository Structure
Code/: preprocessing, training, and analysis scripts
Data/: PD signal datasets
Models/: trained Keras models
Results/: figures used in the manuscript
Execution Order
preprocess.py
train_base_cnn.py
train_transfer_3sensor.py
analyze_attention_weights.py
Requirements
Python >= 3.10
TensorFlow / Keras 3
NumPy, SciPy, Matplotlib, Seaborn
Output
Trained model: cnn_3sensor_transfer_learning.keras
Attention analysis figures in Results/
