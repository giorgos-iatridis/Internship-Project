# Internship-Project

This project explores how to detect synthetic "neutrino-like" pulses in audio signals using wavelet transforms and a simple neural network classifier.

📂 Repository Structure

code.py → Generates synthetic signals with injected pulses and extracts wavelet-based features.

20000&20000.py → Creates 20,000 audio files with pulses and 20,000 without, then saves wavelet coefficients.

TrainingValidationTest.py → Splits the dataset into train / validation / test sets and saves them in .p pickle format.

NeuralNetwork.py → Defines and trains a TensorFlow/Keras model to classify pulse vs. no-pulse signals.

⚙️ Requirements

* numpy
* scipy
* scikit-learn
* matplotlib
* pywavelets
* tensorflow

🚀 Usage

Generate data (requires audio chunks in f1chunks and f2chunks folders):

python 20000&20000.py


Split into datasets:

python TrainingValidationTest.py


Train the neural network:

python NeuralNetwork.py

📊 Results

The network trains but initially struggles to separate pulses from noise.

Likely issue: the scaling factor (C) in pulse generation needs parameter tuning.

🔎 Notes

Raw audio files are not included in this repo.

This project is experimental and intended as a starting point for further exploration.
