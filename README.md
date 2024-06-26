# Security and Privacy --GroupN  
__This repository is the assignment of ELEC0138 Security and Privacy, UCL__   

## 1. DEFINING A THREAT MODEL AND A SECURITY
### Introductions:
Series of significant threats to public transportation systems are caused by the cyber attacks, disrupting operations and compromising passenger safety. These attacks can target infrastructure and IOT devices, or passengers' devices such as signaling systems, camera, speaker, mobile phones, and ticketing platforms, leading to service outages, delays, and potential accidents. By exploiting vulnerabilities in network security, hackers can gain unauthorized access to transportation systems, manipulate schedules, and even hijack aircrafts data remotely. The consequences of the attacks will cause the economic damages, public trust, and harm to passengers.

Since all the attacks are simulated, the traffic will be captured using the Wireshark, they will be saved in pcacp file. These files have been converted to CSV format to facilitate their integration with machine learning in the second part, where these packets are used as the validation dataset.

### Types of attacks:
- DOS
- DDOS
- Spoofing
- Botnet attacks
- Hijacking
- Trojan Horse attacks
All the PCAP files are publically accessible at the Google Drive link: https://drive.google.com/drive/folders/19UdDNUJNECT6fZTOTIUwfYENyGDXqWVU?usp=sharing

Attacks were performed on a range of targets including:
- Websites (Cloned and hosted locally on flux, code included within CW 1)
- IOT devices
- VMs
- Laptops
- Phones

To recreate the attacks, the following tools were used
- DoS/DDoS  [Raven Storm](https://github.com/Tmpertor/Raven-Storm)  [hping3 ](https://www.kali.org/tools/hping3/#:~:text=hping3%20is%20a%20network%20tool,transfer%20files%20under%20supported%20protocols.)
- Spoofing & Hijacking [metasploit](https://github.com/rapid7/metasploit-framework)
- Botnet attacks, Use at your own risk [Mirai](https://github.com/jgamblin/Mirai-Source-Code)

### Formatting the datasets
With the collected PCAPs all the data was formatted to same formated as the CIC IOT dataset through the use of a modified code base provided from  [CICIoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html).

This modification adapted it to run on a Kali VM. The code can be found within CW 1.


## 2. SECURITY/PRIVACY MITIGATIONS
### System: Intrusion Detection and Attack Classification
### Description:
- Utilized datasets from the [CICIoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html) dataset, or generated datasets if needed.
- Trained various ML and DL models for intrusion detection and classification. including RF, CNN, LSTM, GRU, and a hybrid CNN+BiLSTM
- Deployed models on the [Flower Framework](https://flower.ai/) to implement Federated Learning (FL).
- Introduce Local Differential Privacy(LDP) to provide privacy guarantees at the individual user level before any data aggregation or analysis occurs.

### Models:  
- RF: n_estimator=100
- CNN: 2 convolutional layers with 64 filters  
- LSTM: Layer1: 32 units; layer2: 16 units
- CNN + BiLSTM
- GRU
- BiGRU + MLP

### Results:  
1. __Model Test:__  

2. __FL Training:__   
The loss reduced from 0.065 to 0.109, and the accuracy improved from 81.05% to 95.86%.
<img src="CW2/imgs/FLaccloss.png" alt="FLtest" width="700">
3. __Different levels of LDP:__
<img src="CW2/imgs/LDP.png" alt="FLtest" width="500">

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

