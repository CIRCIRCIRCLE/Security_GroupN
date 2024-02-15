# Securing X: a Comprehensive Security and Privacy System

## Model Training
#### 1. Dataset: [CICIoT dataset 2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html)   
 - __originnal traffic:__ in .pcap files, split the data using `TCPDump` into multiple files(2GB each), `Mergecap` can be used to merge the data.   
 - __formatted data:__ extracted from .pcap files into .csv files. All 169 .csv files refer to a combined and shuffled dataset including all attacks. The attacks are identified by the ‘label’ feature. 
#### 2. PreProcessing
- Data Aggregation:  
  - code/data_aggregation
  - aggregate .csv datasets into .pkl files, 80%(135 files) into training set, 20%(34 files) into testing set.   
    _store under datasets/ 'training_data-X_values.pkl(12GB), training_data-y_value.pkl, test_data-X_values.pkl, test_data-y_value.pkl'_
- Data Triming:
  - The training set is over 12GB, which is hard to handle. Do data convert and sampling for the appropriate size.
  - Data convert based on data attributes: float64(46)-->bool(21), float16(2), float32(16), object(1), uint16(3), uint32(3)  12.5GB-->3.9GB 
  - Sampling: the resulting DataFrame contains a balanced representation of each class while reducing the overall number of rows based on the `specified percentage`.
- Define Attack Groups:  attack, category and subcategory labels
    - 2 classes: Benign or Malicious  
    - 8 classes: Benign, DoS, DDoS, Recon, Mirai, Spoofing, Web, BruteForce
    - 34 classes: subgroups of the above 8 classes, details shown in the code and `feature and attacks.pdf` file
- Analysis of attack class distribution

#### 3. ML classifier
- Points from Previous work: [CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment](https://mdpi-res.com/d_attachment/sensors/sensors-23-05941/article_deploy/sensors-23-05941-v2.pdf?version=1687924880) and [wataicyber](https://wataicyber.substack.com/p/comparing-classical-ml-models)
  - Feature selection: 22 important features (duration, srate, drate, syn_flag_number, psh_flag_number, ack_flag_number, ack_count, syn_count, rst_count, header_length, https, ssh, flow_duration, avg, max, tot_sum, min, iat, magnitude, radius, variance)  
  - ML models: Random Forest and Deep Neural Network are able to maintain high accuracy and F-1 score
-  
## Attack Mitagation
