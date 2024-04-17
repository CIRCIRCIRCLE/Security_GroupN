import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from keras.models import load_model
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

"""
Test model : BGRU_MLP_best.h5
             CNN_BLSTM.h5
             CNN.h5
             GRU.h5

Print :     Weighted Precision
            Weighted Recall
            Weighted F1 Score
            Accuracy
            Loss
            Confusion Matrix
            
"""
model = 'BGRU_MLP.h5'
current_directory  = os.path.dirname(__file__)
dataset_path = os.path.join(current_directory, '..', '..', 'datasets')
model_path = os.path.join(current_directory,'..', 'classifiers','model', model)
df = pd.read_csv(os.path.join(dataset_path, 'Test_Set_Filtered.csv'))


# Test Set Filtered
def datasetpreprocess(df):
    X = df.drop(columns=['label'])
    y = df['label']

    # Convert boolean columns to int
    for column in X.columns:
        if X[column].dtype == bool:
            X[column] = X[column].astype(int)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
    print(label_mapping)
    

    # Reshape Data for Conv1D
    X = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
    y = to_categorical(y, num_classes=5)
    
    return X,  y



X,y = datasetpreprocess(df)

model = load_model(model_path)
y_pred_prob = model.predict(X)
y_pred = np.argmax(y_pred_prob, axis=1)
y_test = np.argmax(y, axis=1)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(accuracy)
print(report)
print(df['label'].value_counts())

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)

print("Weighted Precision:", precision)
print("Weighted Recall:", recall)
print("Weighted F1 Score:", f1)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)


plt.figure(figsize=(10, 7))  
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'DDOS', 'DOS', 'Mirai', 'Spoofing'], yticklabels=['Benign', 'DDOS', 'DOS', 'Mirai', 'Spoofing'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()