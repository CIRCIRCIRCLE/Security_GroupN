import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

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


'''
Use CNN for classification
'''
#path direction---------------------------------------------------------
current_directory  = os.path.dirname(__file__)
dataset_path = os.path.join(current_directory, '..', '..', 'datasets')
model_path = os.path.join(current_directory, 'model', 'CNN.h5')
#df2 = pd.read_csv(os.path.join(dataset_path, 'df2.csv'))
df8 = pd.read_csv(os.path.join(dataset_path, 'df8.csv'))
#df34 = pd.read_csv(os.path.join(dataset_path, 'df34.csv'))

#Preprocessing----------------------------------------------------------------
# split features and labels
X = df8.drop(columns=['label'])
y = df8['label']

for column in X.columns:
    if X[column].dtype == bool: X[column] = X[column].astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)  

# Reshape Data
#X_train_reshaped = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
#X_test_reshaped = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

#Build and train the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=256, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5), 
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train Model
history = model.fit(X_train, y_train, epochs=1, batch_size=256, validation_split=0.05)
model.save(model_path)

#Test
model = load_model(model_path)
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
report = classification_report(y_test, y_pred)
print(report)