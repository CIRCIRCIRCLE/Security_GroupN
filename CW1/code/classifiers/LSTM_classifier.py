import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
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
Use LSTM for classification
'''
#path direction---------------------------------------------------------
current_directory  = os.path.dirname(__file__)
dataset_path = os.path.join(current_directory, '..', '..', 'datasets')
model_path = os.path.join(current_directory, 'model', 'LSTM.h5')
#df2 = pd.read_csv(os.path.join(dataset_path, 'df2.csv'))
df8 = pd.read_csv(os.path.join(dataset_path, 'df8.csv'))
#df34 = pd.read_csv(os.path.join(dataset_path, 'df34.csv'))

#Preprocessing----------------------------------------------------------------
# split features and labels
X = df8.drop(columns=['label'])
y = df8['label']

for column in X.columns:
    if X[column].dtype == bool: X[column] = X[column].astype(int)
    #print(f"feature {column}'s data type: {X[column].dtype}" )
# encode labels to numerical representation
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)  
# (3210713, 22) (3210713,) (802679, 22) (802679,)

# convert into 3D tensors (#batch_size, time_steps, seq_len)
X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))  
X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# (3210713, 1, 22) (3210713,) (802679, 1, 22) (802679,)


# Model Setting and training----------------------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=32, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),  #the input represents (num of samples, time step, feature num)
    tf.keras.layers.Dropout(0.5), 
    tf.keras.layers.LSTM(units=16),
    tf.keras.layers.Dropout(0.2), 
    tf.keras.layers.Dense(8, activation='softmax')  # Adjust output units based on your number of classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, epochs=10, batch_size=256, validation_split=0.05)
model.save(model_path)

#Test
model = load_model(model_path)
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
report = classification_report(y_test, y_pred)
print(report)