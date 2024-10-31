#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas scikit-learn')


# In[2]:


get_ipython().system('pip install tensorflow scikit-learn xgboost adversarial-robustness-toolbox')


# In[3]:


import pandas as pd
import os


# In[15]:


unsw_nb15_path = '/Users/kvv2005/Desktop/UNSW_NB15_training-set.csv'


# In[16]:


df= pd.read_csv(unsw_nb15_path)


# In[17]:


print('NSL-KDD Dataset:')
print(nsl_kdd_df.head())


# In[18]:


# Identify the last column as the target label
target_column = df.columns[-1]

# Separate features (all columns except the last one) and target
X = df.iloc[:, :-1]  # All rows, all columns except the last
y = df.iloc[:, -1]   # All rows, only the last column

print(f"Features shape: {X.shape}, Target shape: {y.shape}")


# In[19]:


# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# Drop rows with missing values (if needed)
df_clean = df.dropna()

print(f"New dataset shape after dropping missing values: {df_clean.shape}")


# In[20]:


from sklearn.preprocessing import LabelEncoder

# Identify categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns
print(f"Categorical columns: {categorical_columns}")

# Apply Label Encoding to each categorical column
label_encoders = {}  # Store encoders to decode later if needed
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

print(X.head())


# In[21]:


from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the features
X_scaled = scaler.fit_transform(X)

print(f"Scaled features shape: {X_scaled.shape}")
print('First 5 rows of scaled features:\n', X_scaled[:5])


# In[22]:


import sys
sys.executable


# In[23]:


conda install -c conda-forge keras


# In[24]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

from art.estimators.classification import TensorFlowV2Classifier, SklearnClassifier, XGBoostClassifier
from art.attacks.evasion import HopSkipJump


# In[27]:


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"Training Set: {X_train.shape}, Test Set: {X_test.shape}")


# In[29]:


# Define a simple neural network model
model = Sequential([
    Flatten(input_shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(set(y)), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model on clean data
y_pred_clean = model.predict(X_test).argmax(axis=1)
accuracy_clean = accuracy_score(y_test, y_pred_clean)

print(f"Model Accuracy on Clean Test Data: {accuracy_clean}")
print("Classification Report on Clean Test Data:")
print(classification_report(y_test, y_pred_clean, zero_division=1))


# In[31]:


# Wrap the TensorFlow model with ART's TensorFlowV2Classifier
classifier = TensorFlowV2Classifier(
    model=model,
    nb_classes=len(set(y)),
    input_shape=(X_train.shape[1],),
    loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
)


# In[32]:


from art.attacks.evasion import FastGradientMethod
# Initialize the FGSM attack
fgsm = FastGradientMethod(estimator=classifier, eps=0.1)

# Generate adversarial examples
X_test_adv_fgsm = fgsm.generate(x=X_test[:10])

# Predict on the adversarial examples
y_pred_fgsm = model.predict(X_test_adv_fgsm).argmax(axis=1)

# Evaluate performance on adversarial examples
accuracy_fgsm = accuracy_score(y_test[:10], y_pred_fgsm)
print(f"Model Accuracy on FGSM Adversarial Examples: {accuracy_fgsm}")
print("Classification Report on FGSM Adversarial Examples:")
print(classification_report(y_test[:10], y_pred_fgsm, zero_division=1))


# In[33]:


from art.attacks.evasion import ProjectedGradientDescent
# Initialize the PGD attack
pgd = ProjectedGradientDescent(estimator=classifier, eps=0.1, eps_step=0.01, max_iter=50)

# Generate adversarial examples using PGD
X_test_adv_pgd = pgd.generate(x=X_test[:10])

# Predict on the adversarial examples
y_pred_pgd = model.predict(X_test_adv_pgd).argmax(axis=1)

# Evaluate performance on PGD adversarial examples
accuracy_pgd = accuracy_score(y_test[:10], y_pred_pgd)
print(f"Model Accuracy on PGD Adversarial Examples: {accuracy_pgd}")
print("Classification Report on PGD Adversarial Examples:")
print(classification_report(y_test[:10], y_pred_pgd, zero_division=1))


# In[30]:


# Initialize the C&W attack (L2 norm)
from art.attacks.evasion import CarliniL2Method

cw = CarliniL2Method(classifier=classifier, targeted=False, max_iter=10)

# Generate adversarial examples using C&W
X_test_adv_cw = cw.generate(x=X_test[:10])

# Predict on the adversarial examples
y_pred_cw = model.predict(X_test_adv_cw).argmax(axis=1)

# Evaluate performance on C&W adversarial examples
accuracy_cw = accuracy_score(y_test[:10], y_pred_cw)
print(f"Model Accuracy on C&W Adversarial Examples: {accuracy_cw}")
print("Classification Report on C&W Adversarial Examples:")
print(classification_report(y_test[:10], y_pred_cw, zero_division=1))


# In[31]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Import ART classifiers and attacks
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, CarliniL2Method


# In[32]:


# Assuming X_scaled (features) and y_encoded (labels) are ready
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

print(f"Training Set: {X_train.shape}, Test Set: {X_test.shape}")


# In[33]:


# Define MLP Model
mlp_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(set(y_encoded)), activation='softmax')
])

mlp_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
mlp_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Wrap the model with ART’s TensorFlowV2Classifier
mlp_classifier = TensorFlowV2Classifier(
    model=mlp_model,
    nb_classes=len(set(y_encoded)),
    input_shape=(X_train.shape[1],),
    loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
)


# In[34]:


# FGSM Attack
fgsm = FastGradientMethod(estimator=mlp_classifier, eps=0.1)
X_test_adv_fgsm = fgsm.generate(x=X_test)

# Predictions and Evaluation
y_pred_fgsm = mlp_model.predict(X_test_adv_fgsm).argmax(axis=1)
print("Classification Report (MLP - FGSM):")
print(classification_report(y_test, y_pred_fgsm, zero_division=1))


# In[35]:


# PGD Attack
pgd = ProjectedGradientDescent(estimator=mlp_classifier, eps=0.1, eps_step=0.01, max_iter=50)
X_test_adv_pgd = pgd.generate(x=X_test)

# Predictions and Evaluation
y_pred_pgd = mlp_model.predict(X_test_adv_pgd).argmax(axis=1)
print("Classification Report (MLP - PGD):")
print(classification_report(y_test, y_pred_pgd, zero_division=1))


# In[ ]:


# C&W Attack
cw = CarliniL2Method(classifier=mlp_classifier, targeted=False, max_iter=1)
X_test_adv_cw = cw.generate(x=X_test)

# Predictions and Evaluation
y_pred_cw = mlp_model.predict(X_test_adv_cw).argmax(axis=1)
print("Classification Report (MLP - C&W):")
print(classification_report(y_test, y_pred_cw, zero_division=1))



# In[34]:


# Reshape input for CNN
X_train_cnn = X_train[..., None]
X_test_cnn = X_test[..., None]

# Define CNN Model
cnn_model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(set(y_encoded)), activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Wrap the model with ART’s TensorFlowV2Classifier
cnn_classifier = TensorFlowV2Classifier(
    model=cnn_model,
    nb_classes=len(set(y_encoded)),
    input_shape=(X_train.shape[1], 1),
    loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
)


# In[160]:


# FGSM Attack
fgsm = FastGradientMethod(estimator=cnn_classifier, eps=0.1)
X_test_adv_fgsm_cnn = fgsm.generate(x=X_test_cnn)

# Predictions and Evaluation
y_pred_fgsm_cnn = cnn_model.predict(X_test_adv_fgsm_cnn).argmax(axis=1)
print("Classification Report (CNN - FGSM):")
print(classification_report(y_test, y_pred_fgsm_cnn, zero_division=1))


# In[162]:


# PGD Attack
pgd = ProjectedGradientDescent(estimator=cnn_classifier, eps=0.1, eps_step=0.01, max_iter=50)
X_test_adv_pgd_cnn = pgd.generate(x=X_test_cnn)

# Predictions and Evaluation
y_pred_pgd_cnn = cnn_model.predict(X_test_adv_pgd_cnn).argmax(axis=1)
print("Classification Report (CNN - PGD):")
print(classification_report(y_test, y_pred_pgd_cnn, zero_division=1))


# In[37]:


# C&W Attack
cw = CarliniL2Method(classifier=cnn_classifier, targeted=False, max_iter=10)
X_test_adv_cw_cnn = cw.generate(x=X_test_cnn)

# Predictions and Evaluation
y_pred_cw_cnn = cnn_model.predict(X_test_adv_cw_cnn).argmax(axis=1)
print("Classification Report (CNN - C&W):")
print(classification_report(y_test, y_pred_cw_cnn, zero_division=1))


# In[ ]:




