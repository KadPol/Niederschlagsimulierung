# Comparison of streamflow modelling of choosen citites between a standard Long Short-Term Memory Network (LSTM) and a combined Convolutional Neuronal Network with Long Short-Term Memory Network (CNNLSTM)

### This notebooks reproduces the results related to the abstract "Machine learning-based time-series prediction for rainfall-runoff modeling"

### Polat, Kadir and Salhi, Hamza: "Machine learning-based time-series prediction for rainfall-runoff modeling"

### Find the paper here: https://medium.com/@p.kadir/machine-learning-based-time-series-prediction-for-rainfall-runoff-modeling-604d0760392e

# Get Started
### To run the code without any problems consider following steps.

### 1) Create an environment to install the packages definded in requirements.txt. It's on you whether you create the environment with conda or Venv.

### 2) Download the Camels-US Dataset and Daymet-US Dataset to predict a defined basin. To downlaod teh data see following webpages: Camels US Dataset: https://ral.ucar.edu/solutions/products/camels and Daymet US Dataset: https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1840 

### 3) Copy the Daymet dataset to a folder named 'data_daymet' and the Camels dataset to a folder named 'data'. For Camels dataset streamflow and dataset of each basin can be copied directly to the folder. For the daymet dataset only streamflow can be copied directly to the folder. For each basin an additional folder shoul be created in which the dataset can be copied. 

# Project structure

### ./data: whole camels dataset for all basins is in this folder streamflow dataset included.

### ./data_daymet: whole daymet dataset for all basins is in this folder streamflow dataset is included. Each basin is saved in a separeted folder. For example: ./data/9494000: with the specific number of Mesa.

### ./scripts: all addional external function needed for the notebook can be find in scripts. In this case dataio, getData, preProcess, model and postProcess files are included in scripts folder.

# Run the Code

### For reproducing the figures you can run the Streamflow_prediction.ipynb via Jupyter Notebook. Choose your needed environment and start the server by running jupyter notebook.

# License

### Apache License, Version 2.0
