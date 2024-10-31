#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install pandas scikit-learn')


# In[5]:


import pandas as pd
import os


# In[6]:


nsl_kdd_path = '/Users/marlenawasiak/Desktop/Data_Collection/NSL_KDD_Train.csv'


# In[7]:


nsl_kdd_df= pd.read_csv(nsl_kdd_path)


# In[12]:


print('NSL-KDD Dataset:')
print(nsl_kdd_df.head())


# In[14]:


print("Missing values:")
print(nsl_kdd_df.isnull().sum())


# In[16]:


print(nsl_kdd_df.columns)


# In[18]:


x= nsl_kdd_df.drop('normal', axis= 1)
y= nsl_kdd_df['normal']


# In[20]:


print('Features shape X: ', x.shape)
print('Features shape Y:', y.shape)


# In[22]:


x.head(),y.head()


# In[24]:


from sklearn.preprocessing import LabelEncoder
label_encoder= LabelEncoder()
y_encoded= label_encoder.fit_transform(y)

print('First 5 encoded labels:', y_encoded[:5])
print('Label classes:', label_encoder.classes_)


# In[26]:


non_numeric_columns= x.select_dtypes(include=['object']).columns
X_encoded= pd.get_dummies(x, columns=non_numeric_columns)
print(X_encoded.head())
#print(x.dtypes)


# In[28]:


from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()
X_scaled= scaler.fit_transform(X_encoded)
print('First 5 rows of scaled features:\n', X_scaled[:5])


# In[30]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
get_ipython().system('pip install xgboost')
import xgboost as xgb
import numpy as np


# In[31]:


# Split the scaled feature set and encoded labels into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
unique_classes, class_counts = np.unique(y_train, return_counts=True)
class_weights = {cls: max(class_counts) / count for cls, count in zip(unique_classes, class_counts)}

# Check the shapes of the training and testing sets
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)


# In[34]:


import numpy as np

# Check the distribution of labels in the training set
unique, counts = np.unique(y_train, return_counts=True)
print("Training set label distribution:", dict(zip(unique, counts)))

# Check the distribution of labels in the test set
unique, counts = np.unique(y_test, return_counts=True)
print("Test set label distribution:", dict(zip(unique, counts)))


# In[36]:


rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf, zero_division=1))


# In[37]:


# Train the XGBoost model with class weights
# Create sample weights based on the class of each sample in the training set
sample_weights = np.array([class_weights[label] for label in y_train])

xgb_model = xgb.XGBClassifier(n_estimators=100, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the XGBoost model
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb, zero_division=1))


# In[40]:


#Train the SVM model with class weights
# Import SVM (SVC) and other necessary libraries
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

svm_model = SVC(kernel='linear', class_weight='balanced', random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Evaluate the SVM model
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm, zero_division=1))


# In[42]:


get_ipython().system('pip install adversarial-robustness-toolbox')


# In[44]:


from art.estimators.classification import SklearnClassifier

# Wrap all three models with ART's SklearnClassifier
rf_art_classifier = SklearnClassifier(model=rf_model)    # Random Forest
svm_art_classifier = SklearnClassifier(model=svm_model)  # SVM


# In[46]:


from art.attacks.evasion import BoundaryAttack

# Initialize the Boundary Attack for each model
boundary_attack_rf = BoundaryAttack(estimator=rf_art_classifier, targeted=False, max_iter=100)
boundary_attack_svm = BoundaryAttack(estimator=svm_art_classifier, targeted=False, max_iter=100)


# Select a small subset of the test data to run the attack
X_test_sample = X_test[:10]
y_test_sample = y_test[:10]

# Generate adversarial examples for all models
X_test_adv_rf = boundary_attack_rf.generate(x=X_test_sample, y=y_test_sample)
X_test_adv_svm = boundary_attack_svm.generate(x=X_test_sample, y=y_test_sample)


# In[48]:


from sklearn.metrics import accuracy_score, classification_report

# Evaluate Random Forest
y_pred_rf = rf_model.predict(X_test_adv_rf)
accuracy_rf = accuracy_score(y_test_sample, y_pred_rf)
print(f"Random Forest Accuracy on Boundary Attack Adversarial Examples: {accuracy_rf}")
print("Random Forest Classification Report:")
print(classification_report(y_test_sample, y_pred_rf, zero_division=1))

# Evaluate SVM
y_pred_svm = svm_model.predict(X_test_adv_svm)
accuracy_svm = accuracy_score(y_test_sample, y_pred_svm)
print(f"SVM Accuracy on Boundary Attack Adversarial Examples: {accuracy_svm}")
print("SVM Classification Report:")
print(classification_report(y_test_sample, y_pred_svm, zero_division=1))


# In[50]:


from art.attacks.evasion import HopSkipJump

# Initialize HopSkipJump Attack for each model
hopskip_attack_rf = HopSkipJump(classifier=rf_art_classifier, targeted=False, max_iter=50)
hopskip_attack_svm = HopSkipJump(classifier=svm_art_classifier, targeted=False, max_iter=50)

# Generate adversarial examples for all models using HopSkipJump Attack
X_test_adv_rf_hop = hopskip_attack_rf.generate(x=X_test_sample)
X_test_adv_svm_hop = hopskip_attack_svm.generate(x=X_test_sample)


# In[52]:


# Evaluate Random Forest on HopSkipJump Attack
y_pred_rf_hop = rf_model.predict(X_test_adv_rf_hop)
accuracy_rf_hop = accuracy_score(y_test_sample, y_pred_rf_hop)
print(f"Random Forest Accuracy on HopSkipJump Attack Adversarial Examples: {accuracy_rf_hop}")
print("Random Forest Classification Report (HopSkipJump):")
print(classification_report(y_test_sample, y_pred_rf_hop, zero_division=1))

# Evaluate SVM on HopSkipJump Attack
y_pred_svm_hop = svm_model.predict(X_test_adv_svm_hop)
accuracy_svm_hop = accuracy_score(y_test_sample, y_pred_svm_hop)
print(f"SVM Accuracy on HopSkipJump Attack Adversarial Examples: {accuracy_svm_hop}")
print("SVM Classification Report (HopSkipJump):")
print(classification_report(y_test_sample, y_pred_svm_hop, zero_division=1))


# In[60]:


from art.estimators.classification import SklearnClassifier, XGBoostClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, HopSkipJump
from art.attacks.evasion import BoundaryAttack


# In[56]:


# Manually set the input shape (number of features in your data)
input_shape = (X_train.shape[1],)

# Wrap the XGBoost model with ART's XGBoostClassifier
xgb_art_classifier = XGBoostClassifier(model=xgb_model, clip_values=(0, 1), nb_features=input_shape[0])


# In[62]:


# Initialize the HopSkipJump attack
hopskip_attack = HopSkipJump(classifier=xgb_art_classifier, targeted=False, max_iter=50)

# Select a small subset of the test data
X_test_sample = X_test[:10]

# Generate adversarial examples using HopSkipJump
X_test_adv_hop = hopskip_attack.generate(x=X_test_sample)

# Predict on the adversarial examples
y_pred_hop = xgb_model.predict(X_test_adv_hop)

# Evaluate the model on HopSkipJump adversarial examples
accuracy_hop = accuracy_score(y_test[:10], y_pred_hop)
print(f"XGBoost Accuracy on HopSkipJump Adversarial Examples: {accuracy_hop}")

# Display classification report
print("Classification Report (XGBoost - HopSkipJump):")
print(classification_report(y_test[:10], y_pred_hop, zero_division=1))



# In[64]:


# Initialize the Boundary Attack
boundary_attack = BoundaryAttack(estimator=xgb_art_classifier, targeted=False, max_iter=50)

# Select a small subset of the test data
X_test_sample = X_test[:10]

# Generate adversarial examples using Boundary Attack
X_test_adv_boundary = boundary_attack.generate(x=X_test_sample)

# Predict on the adversarial examples
y_pred_boundary = xgb_model.predict(X_test_adv_boundary)

# Evaluate the model on Boundary Attack adversarial examples
accuracy_boundary = accuracy_score(y_test[:10], y_pred_boundary)
print(f"XGBoost Accuracy on Boundary Attack Adversarial Examples: {accuracy_boundary}")

# Display classification report
print("Classification Report (XGBoost - Boundary Attack):")
print(classification_report(y_test[:10], y_pred_boundary, zero_division=1))


# In[66]:


get_ipython().system('pip install shap scikit-learn xgboost')


# In[68]:


import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# In[83]:


# Initialize SHAP KernelExplainer for SVM
explainer_svm = shap.KernelExplainer(svm_model.predict_proba, X_train[:100])

# Generate SHAP values for the first 10 samples of the test set
shap_values_svm = explainer_svm.shap_values(X_test[:10])

# Plot SHAP summary plot for SVM
shap.summary_plot(shap_values_svm, X_test[:10])


# In[79]:


# Initialize SHAP TreeExplainer for Random Forest
explainer_rf = shap.TreeExplainer(rf_model)

# Generate SHAP values for the first 10 samples of the test set
shap_values_rf = explainer_rf.shap_values(X_test[:10])

# Plot SHAP summary plot for Random Forest
shap.summary_plot(shap_values_rf, X_test[:10])


# In[86]:


# Initialize SHAP TreeExplainer for XGBoost
explainer_xgb = shap.TreeExplainer(xgb_model)

# Generate SHAP values for the first 10 samples of the test set
shap_values_xgb = explainer_xgb.shap_values(X_test[:10])

# Plot SHAP summary plot for XGBoost
shap.summary_plot(shap_values_xgb, X_test[:10])


# In[90]:


# SHAP summary plot for clean data (XGBoost)
shap.summary_plot(shap_values_xgb, X_test[:10], title="XGBoost - Clean Data")

# SHAP summary plot for clean data (Random Forest)
shap.summary_plot(shap_values_rf, X_test[:10], title="Random Forest - Clean Data")

# SHAP summary plot for clean data (SVM)
shap.summary_plot(shap_values_svm, X_test[:10], title="SVM - Clean Data")


# In[98]:


# Initialize SHAP explainer for SVM
explainer_svm = shap.KernelExplainer(svm_model.predict_proba, X_train[:100])

# SHAP values for Boundary Attack on SVM
shap_values_boundary_svm = explainer_svm.shap_values(X_test_adv_svm)
shap.summary_plot(shap_values_boundary_svm, X_test_adv_svm, title="SVM - Boundary Attack")

# SHAP values for HopSkipJump Attack on SVM
shap_values_hop_svm = explainer_svm.shap_values(X_test_adv_svm_hop)
shap.summary_plot(shap_values_hop_svm, X_test_adv_svm_hop, title="SVM - HopSkipJump Attack")


# In[104]:


# Initialize SHAP explainer for Random Forest
explainer_rf = shap.TreeExplainer(rf_model)

# SHAP values for Boundary Attack on Random Forest
shap_values_boundary_rf = explainer_rf.shap_values(X_test_adv_rf)
shap.summary_plot(shap_values_boundary_rf, X_test_adv_rf, title="Random Forest - Boundary Attack")

# SHAP values for HopSkipJump Attack on Random Forest
shap_values_hop_rf = explainer_rf.shap_values(X_test_adv_rf_hop)
shap.summary_plot(shap_values_hop_rf, X_test_adv_rf_hop, title="Random Forest - HopSkipJump Attack")


# In[106]:


# Initialize SHAP explainer for XGBoost
explainer_xgb = shap.TreeExplainer(xgb_model)

# SHAP values for Boundary Attack on XGBoost
shap_values_boundary_xgb = explainer_xgb.shap_values(X_test_adv_boundary)
shap.summary_plot(shap_values_boundary_xgb,X_test_adv_boundary, title="XGBoost - Boundary Attack")

# SHAP values for HopSkipJump Attack on XGBoost
shap_values_hop_xgb = explainer_xgb.shap_values(X_test_adv_hop)
shap.summary_plot(shap_values_hop_xgb, X_test_adv_hop, title="XGBoost - HopSkipJump Attack")


# In[ ]:




