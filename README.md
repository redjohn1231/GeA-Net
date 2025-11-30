# GeA-Net
Structure Searchable Network Model for Deepfake Image Detection

This repository contains the implementation of a deepfake detection method with automatic network structure search based on genetic algorithms.

## Repository Structure

The repository is organized into two main folders: `Network structure search` and `Deepfake detection`.

### Network structure search
This folder contains scripts for automatically searching and optimizing the network structure.

- **Network_builder_1.py**  
  Constructs a network based on an input code, trains the network, and returns its fitness value.

- **function.py**  
  Contains all utility functions used for the genetic algorithm, including mutation, crossover, and selection operations.

- **network_search.py**  
  The main program for network structure search using the genetic algorithm.

- **Image_preprocess.py**  
  Implements the data generator for loading and preprocessing the training and validation images.

- **Setting.py**  
  Contains dataset paths and other configuration settings required for training and search.

### Deepfake detection
This folder contains scripts for training the networks after structure search.

- **Train.py**  
  The main training script for the deepfake detection network.

- **Image_preprocess.py**  
  Data generator for training and validation datasets.

- **Network_builder_2.py**  
  Constructs a network based on a code, trains it, and saves the trained model.

## Requirements

- **Python**: 3.9 
- **TensorFlow**: 2.9.1  
- **Keras**: 2.9.0 (included in TensorFlow 2.9.1)  
- **NumPy**: 1.19.5  
- **Matplotlib**: 3.8.4  
- **OpenCV**: 4.12.0.88  
