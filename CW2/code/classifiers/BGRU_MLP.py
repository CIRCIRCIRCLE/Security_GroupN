import tensorflow as tf
from keras.models import Sequential
from keras.layers import Bidirectional, GRU, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import load_model
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print('GPU is successfully loaded')
print('-------------------------------------')

current_directory  = os.path.dirname(__file__)
dataset_path = os.path.join(current_directory, '..', '..', 'datasets')
model_path = os.path.join(current_directory, 'model', 'BGRU_MLP.h5')
df = pd.read_csv(os.path.join(dataset_path, 'filtered_df.csv'))


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
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)  

    # Reshape Data for Conv1D
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    y_train = to_categorical(y_train, num_classes=5)
    y_test = to_categorical(y_test, num_classes=5)

    return X_train_reshaped, X_test_reshaped, y_train, y_test

gru_units=128
mlp_layers=2
mlp_units=128
dropout_rate=0.4
num_classes = 5

def build_BGRU_MLP_classifier(X_train, y_train):
    model = Sequential([
        # Bidirectional GRU layer
        Bidirectional(GRU(gru_units, return_sequences=False)),
        
        # MLP Hidden layers
        Dense(mlp_units, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(mlp_units, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])

    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size = 32,validation_data=(X_test, y_test))
    return model

X_train, X_test, y_train, y_test = datasetpreprocess(df)


# Create Model
model = build_BGRU_MLP_classifier(X_train,y_train)
model.summary()  
model.save(model_path)

# Test
model = load_model(model_path)
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_test = np.argmax(y_test, axis=1)

# Calculation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)

print(report)
print(df['label'].value_counts())
print("Weighted Precision:", precision)
print("Weighted Recall:", recall)
print("Weighted F1 Score:", f1)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)
