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
- RF
- CNN
- LSTM
- CNN + BiLSTM
- GRU

### Results:  
1. __Model Test:__  
During model training, various features were extracted from network flow data.   
These models achieved exceptional performance with accuracy, precision, recall, and F1-score all exceeding 99%.
2. __FL Training:__   
The loss reduced from 0.065 to 0.109, and the accuracy improved from 81.05% to 95.86%.

### Code:
