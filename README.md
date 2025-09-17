# ThermoONet_Comet: A Deep Learning Framework for Cometary Thermophysical Modeling and Parameter Inversion
ThermoONet is a PyTorch-based deep learning framework designed for cometary thermophysical modeling and parameter inversion. It uses a modified deep operator neural network architecture with channel attention mechanisms to model the thermal behavior of comet nuclei and estimate physical parameters from observational data.
## Features
* **Thermophysical Modeling**: Predicts subsurface temperature distributions and water production rates for cometary nuclei
* **Size Inversion**: Estimates nucleus size of comets using water production data from observations
* **Custom Architecture**: Implements channel attention mechanisms and branch networks for specialized feature processing
* **Multi-branch Design**: Processes different types of input parameters through specialized network branches
## Overview
This repository contains three main components:
1. **ThermoONet Architecture**: The modified deep operator neural network architecture with channel attention mechanisms
2. **67P/Churyumov-Gerasimenko Benchmark**: Application of ThermoONet to model the thermal behavior of comet 67P
3. **Comet Size Inversion**: Implementation for estimating nucleus size of comet C/2002 Y1 (Juels-Holvorcem) through the water production rate provided by SOHO/SWAN observations
## Prerequisites
* Python 3.8+
* PyTorch (GPU version recommended)
* NumPy
* SciPy
* Matplotlib
* pandas
* plyfile 
...
## GPU Support
For optimal performance, install the PyTorch version that matches your GPU capabilities. Please refer to the [PyTorch official website](https://pytorch.org/) for installation instructions specific to your hardware.
## Usage
## 1. Neural Network Architecture (ThermoONet_architecture.py)
This file contains the core neural network architecture including:
* SELayer: Channel attention mechanism module
* Branch1, Branch2, Branch3: Specialized branch networks
* Branch: Main network that combines all branches
## 2. 67P Benchmark Testing ([Test_67P.py](Test_67P/Test_ThermoONet_67P.py))
Applies ThermoONet to comet 67P/Churyumov-Gerasimenko:









