import pandas as pd
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder

# Read the CSV file
ADCTL_data = pd.read_csv('ADCTL_Result.csv')

# Extract the predicted and actual labels
ADCTL_pred_labels = ADCTL_data['Pred_Label']
ADCTL_actual_labels = ADCTL_data['Actual_Label']

# Encode labels to numeric values
label_encoder = LabelEncoder()
ADCTL_pred_labels = label_encoder.fit_transform(ADCTL_pred_labels)
ADCTL_actual_labels = label_encoder.transform(ADCTL_actual_labels)

# Calculate AUC score
ADCTL_auc_score = roc_auc_score(ADCTL_actual_labels, ADCTL_pred_labels)
print('ADCTL AUC Score:', ADCTL_auc_score)

# Calculate MCC score
ADCTL_mcc_score = matthews_corrcoef(ADCTL_actual_labels, ADCTL_pred_labels)
print('ADCTL MCC Score:', ADCTL_mcc_score)


# Read the CSV file
ADMCI_data = pd.read_csv('ADMCI_Result.csv')

# Extract the predicted and actual labels
ADMCI_pred_labels = ADMCI_data['Pred_Label']
ADMCI_actual_labels = ADMCI_data['Actual_Label']

# Encode labels to numeric values
label_encoder = LabelEncoder()
ADMCI_pred_labels = label_encoder.fit_transform(ADMCI_pred_labels)
ADMCI_actual_labels = label_encoder.transform(ADMCI_actual_labels)

# Calculate AUC score
ADMCI_auc_score = roc_auc_score(ADMCI_actual_labels, ADMCI_pred_labels)
print('ADMCI AUC Score:', ADMCI_auc_score)

# Calculate MCC score
ADMCI_mcc_score = matthews_corrcoef(ADMCI_actual_labels, ADMCI_pred_labels)
print('ADMIC MCC Score:', ADMCI_mcc_score)


# Read the CSV file
MCICTL_data = pd.read_csv('MCICTL_Result.csv')

# Extract the predicted and actual labels
MCICTL_pred_labels = MCICTL_data['Pred_Label']
MCICTL_actual_labels = MCICTL_data['Actual_Label']

# Encode labels to numeric values
label_encoder = LabelEncoder()
MCICTL_pred_labels = label_encoder.fit_transform(MCICTL_pred_labels)
MCICTL_actual_labels = label_encoder.transform(MCICTL_actual_labels)

# Calculate AUC score
MCICTL_auc_score = roc_auc_score(MCICTL_actual_labels, MCICTL_pred_labels)
print('MCICTL AUC Score:', MCICTL_auc_score)

# Calculate MCC score
MCICTL_mcc_score = matthews_corrcoef(MCICTL_actual_labels, MCICTL_pred_labels)
print('MCICTL MCC Score:', MCICTL_mcc_score)
