# Security and Privacy --GroupN  
__This repository is the assignment of ELEC0138 Security and Privacy, UCL__   

## 1. DEFINING A THREAT MODEL AND A SECURITY


## 2. SECURITY/PRIVACY MITIGATIONS
### System: Intrusion Detection and Attack Classification
### Description:
- Utilized datasets from the [CICIoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html) dataset, or generated datasets if needed.
- Trained various ML and DL models for intrusion detection and classification. including RF, CNN, LSTM, GRU, and a hybrid CNN+BiLSTM
- Deployed models on the [Flower Framework](https://flower.ai/) to implement Federated Learning (FL).

### Models:  
- RF: n_estimator=100
- CNN:   
  <img src="CW2/imgs/CNN.png" alt="CNN" width="500">
- LSTM:   
  <img src="CW2/imgs/LSTM.png" alt="LSTM" width="400">
- CNN + BiLSTM
- GRU

### Results:  
1. __Model Test:__  
During model training, various features were extracted from network flow data.   
These models achieved exceptional performance with accuracy, precision, recall, and F1-score all exceeding 99%.
2. __FL Training:__   
The loss reduced from 0.065 to 0.109, and the accuracy improved from 81.05% to 95.86%.
<img src="CW2/imgs/FLaccloss.png" alt="FLtest" width="700">

### Code Instruction:   
__Data Manipulation:__
```python
# Aggregate the data and reduce the package size
python CW2/code/data manipulation/data_aggregation.py

# Group the data: eg. 'DDoS-SYN_Flood'->'DDoS', 'DNS_Spoofing'->'Spoofing'
python CW2/code/data manipulation/data_group_and_eda.py

# filter the needed classes and sample a balanced data, the processed data will be stored under /datasets folder  
python CW2/code/data manipulation/data_filter.py
```
__Train the Model:__
```
python CW2/code/classifiers/RF_classifier.py   
python CW2/code/classifiers/CNN_classifier.py
python CW2/code/classifiers/LSTM_classifier.py
```

__Train the Model through FL:__  
```python
# First run the server, set the server address correctly (default: 0.0.0.0:8080)
python CW2/code/FL/server.py

# Then run the client, at least 2 clients to start training.
# (default connected server address: 127.0.0.1:8080)
python CW2/code/FL/client.py
```

Simulation can be accessed here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kbpPHU2B1tlXQX1mixUxZ4PE2nTpAJWm?usp=sharing)

