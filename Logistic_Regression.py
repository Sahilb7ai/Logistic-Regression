import os
import pandas as pd
import numpy as np

os.chdir("C:/Users/Acer/OneDrive/Desktop/log/")

FullRaw = pd.read_csv('BankCreditCard.csv')


############################
# Sampling: Divide the data into Train and Testset
############################

from sklearn.model_selection import train_test_split
TrainRaw, TestRaw = train_test_split(FullRaw, train_size=0.7, random_state = 123)


# Create Source Column in both Train and Test
TrainRaw['Source'] = 'Train'
TestRaw['Source'] = 'Test'

# Combine Train and Test
FullRaw = pd.concat([TrainRaw, TestRaw], axis = 0)
FullRaw.shape

# Check for NAs
FullRaw.isnull().sum()

# % Split of 0s and 1s
FullRaw.loc[FullRaw['Source'] == 'Train', 'Default_Payment'].value_counts()/ FullRaw[FullRaw['Source'] == 'Train'].shape[0]


# Summarize the data
FullRaw_Summary = FullRaw.describe()
# FullRaw_Summary = FullRaw.describe(include = "all")

# Lets drop 'Customer ID' column from the data as it is not going to assist us in our model
FullRaw.drop(['Customer ID'], axis = 1, inplace = True) 
FullRaw.shape

############################
# Use data description excel sheet to convert numeric variables to categorical variables
############################

# Categorical variables: Gender, Academic_Qualification, Marital

Variable_To_Update = 'Gender'
FullRaw[Variable_To_Update].value_counts() # To check the unique categories of the variable
FullRaw[Variable_To_Update].replace({1:"Male", 
                                     2:"Female"}, inplace = True)
FullRaw[Variable_To_Update].value_counts()


Variable_To_Update = 'Academic_Qualification'
FullRaw[Variable_To_Update].value_counts()
FullRaw[Variable_To_Update].replace({1:"Undergraduate",
                                     2:"Graduate",
                                     3:"Postgraduate",
                                     4:"Professional",
                                     5:"Others",
                                     6:"Unknown"}, inplace = True)
FullRaw[Variable_To_Update].value_counts()


Variable_To_Update = 'Marital'
FullRaw[Variable_To_Update].value_counts()
FullRaw[Variable_To_Update].replace({1:"Married",
                                     2:"Single",
                                     3:"Unknown",
                                     0:"Unknown"}, inplace = True)
FullRaw[Variable_To_Update].value_counts()


############################
# Dummy variable creation
############################

FullRaw2 = pd.get_dummies(FullRaw, drop_first = True) # 'Source'  column will change to 'Source_Train' and it contains 0s and 1s
FullRaw2.shape


############################
# Divide the data into Train and Test
############################

# Step 1: Divide into Train and Testest
Train = FullRaw2[FullRaw2['Source_Train'] == 1].drop(['Source_Train'], axis = 1).copy()
Test = FullRaw2[FullRaw2['Source_Train'] == 0].drop(['Source_Train'], axis = 1).copy()


# Step 2: Divide into Xs (Indepenedents) and Y (Dependent)
depVar = 'Default_Payment'
trainX = Train.drop([depVar], axis = 1).copy()
trainY = Train[depVar].copy()
testX = Test.drop([depVar], axis = 1).copy()
testY = Test[depVar].copy()

trainX.shape
testX.shape

############################
# Add Intercept Column
############################

# In Python, linear regression function does NOT account for an intercept.
# So, we need to specify a column which has a constant value of 1 
from statsmodels.api import add_constant
trainX = add_constant(trainX)
testX = add_constant(testX)

trainX.shape
testX.shape

#########################
# VIF check
#########################
from statsmodels.stats.outliers_influence import variance_inflation_factor

tempMaxVIF = 10 # This VIF variable will be calculated at EVERY iteration in the while loop
maxVIF = 10
trainXCopy = trainX.copy()
counter = 1
highVIFColumnNames = []

while (tempMaxVIF >= maxVIF):
    
    print(counter)
    
    # Create an empty temporary df to store VIF values
    tempVIFDf = pd.DataFrame()
    
    # Calculate VIF using list comprehension
    tempVIFDf['VIF'] = [variance_inflation_factor(trainXCopy.values, i) for i in range(trainXCopy.shape[1])]
    
    # Create a new column "Column_Name" to store the col names against the VIF values from list comprehension
    tempVIFDf['Column_Name'] = trainXCopy.columns
    
    # Drop NA rows from the df - If there is some calculation error resulting in NAs
    tempVIFDf.dropna(inplace=True)
    
    # Sort the df based on VIF values, then pick the top most column name (which has the highest VIF)
    tempColumnName = tempVIFDf.sort_values(["VIF"], ascending = False).iloc[0,1]
    
    # Store the max VIF value in tempMaxVIF
    tempMaxVIF = tempVIFDf.sort_values(["VIF"], ascending = False).iloc[0,0]
    
    if (tempMaxVIF >= maxVIF): # This condition will ensure that columns having VIF lower than 5 are NOT dropped
        
        # Remove the highest VIF valued "Column" from trainXCopy. As the loop continues this step will keep removing highest VIF columns one by one 
        trainXCopy = trainXCopy.drop(tempColumnName, axis = 1)    
        highVIFColumnNames.append(tempColumnName)
        print(tempColumnName)
    
    counter = counter + 1

highVIFColumnNames


highVIFColumnNames.remove('const') # We need to exclude 'const' column from getting dropped/ removed. This is intercept.
highVIFColumnNames

trainX = trainX.drop(highVIFColumnNames, axis = 1)
testX = testX.drop(highVIFColumnNames, axis = 1)


trainX.shape
testX.shape



########################
# Model building
########################

# Build logistic regression model (using statsmodels package/library)
from statsmodels.api import Logit  
M1 = Logit(trainY, trainX) # (Dep_Var, Indep_Vars) # This is model definition
M1_Model = M1.fit() # This is model building
M1_Model.summary() # This is model output/summary

# M1_Model = Logit(trainY, trainX).fit()
# M1_Model.summary()

# Some interpretation
# Log-Likelihood: # Higher the "Log-Likelihood" value, better the model
# Pseudo R-squ.: [1 - (Log-Likelihood/LL-Null)] 
# Higher the Pseudo R-sq. value better the model
# AIC = -2(Log-Likelihood) + 2K, where K is total number of indep variables
# Lower the AIC value better the model

# 0.8832 is the Repayment_Status_Jan Coeff, 0.030 is the Std Error (Found in model output table) 
# 2.5% Up and Down (5% in total) for Repayment_Status_Jan Coeff (0.8832) is calculated as:
# Lower 2.5% is (0.8832 - 1.96 * 0.030) -> 0.824 & Upper 2.5% is (0.8832 + 1.96 * 0.030) -> 0.942
# 1.96 is the 95% Confidence Interval (CI)


########################
# Manual model selection. Drop the most insignificant variable in model one by one and recreate the model
########################

# Drop Marital_Unknown
Cols_To_Drop = ["Marital_Unknown"]
M2 = Logit(trainY, trainX.drop(Cols_To_Drop, axis = 1)).fit() # (Dep_Var, Indep_Vars)
M2.summary()


# Drop April_Bill_Amount
Cols_To_Drop.append('April_Bill_Amount')
M3 = Logit(trainY, trainX.drop(Cols_To_Drop, axis = 1)).fit() # (Dep_Var, Indep_Vars)
M3.summary()

# Drop June_Bill_Amount
Cols_To_Drop.append('June_Bill_Amount')
M5 = Logit(trainY, trainX.drop(Cols_To_Drop, axis = 1)).fit() # (Dep_Var, Indep_Vars)
M5.summary()

# Drop Academic_Qualification_Postgraduate
Cols_To_Drop.append('Academic_Qualification_Postgraduate')
M6 = Logit(trainY, trainX.drop(Cols_To_Drop, axis = 1)).fit() # (Dep_Var, Indep_Vars)
M6.summary()

# Drop Age_Years
Cols_To_Drop.append('Age_Years')
M7 = Logit(trainY, trainX.drop(Cols_To_Drop, axis = 1)).fit() # (Dep_Var, Indep_Vars)
M7.summary()

# Drop Repayment_Status_Feb
Cols_To_Drop.append('Repayment_Status_Feb')
M8 = Logit(trainY, trainX.drop(Cols_To_Drop, axis = 1)).fit() # (Dep_Var, Indep_Vars)
M8.summary()

# Drop Academic_Qualification_Unknown
Cols_To_Drop.append('Academic_Qualification_Unknown')
M9 = Logit(trainY, trainX.drop(Cols_To_Drop, axis = 1)).fit() # (Dep_Var, Indep_Vars)
M9.summary()

# Drop Academic_Qualification_Undergraduate
Cols_To_Drop.append('Academic_Qualification_Undergraduate')
M10 = Logit(trainY, trainX.drop(Cols_To_Drop, axis = 1)).fit() # (Dep_Var, Indep_Vars)
M10.summary()

# Drop Previous_Payment_May
Cols_To_Drop.append('Previous_Payment_May')
M11 = Logit(trainY, trainX.drop(Cols_To_Drop, axis = 1)).fit() # (Dep_Var, Indep_Vars)
M11.summary()

# Drop Repayment_Status_April
Cols_To_Drop.append('Repayment_Status_April')
M11 = Logit(trainY, trainX.drop(Cols_To_Drop, axis = 1)).fit() # (Dep_Var, Indep_Vars)
M11.summary()

############################
# Prediction and Validation
############################

trainX = trainX.drop(Cols_To_Drop, axis = 1)
testX = testX.drop(Cols_To_Drop, axis = 1) 

trainX.shape
testX.shape



testX['Test_Prob'] = M11.predict(testX) # Store probability predictions in "Text_X" df
testX.columns # A new column called Test_Prob should be created
testX['Test_Prob'][0:6]
testY[:6]

# Classify 0 or 1 based on 0.5 cutoff
testX['Test_Class'] = np.where(testX['Test_Prob'] >= 0.5, 1, 0)
testX.columns # A new column called Test_Class should be created


########################
# Confusion matrix
########################

Confusion_Mat = pd.crosstab(testX['Test_Class'], testY) # R, C format
Confusion_Mat

# Check the accuracy of the model
(sum(np.diagonal(Confusion_Mat))/testX.shape[0])*100 # ~82%

########################
# Precision, Recall, F1 Score
########################

from sklearn.metrics import classification_report
print(classification_report(testY, testX['Test_Class'])) # Actual, Predicted

# Precision: TP/(TP+FP) # [TP/Total Predicted Positives]
# Recall: TP/(TP+FN) # [TP/Total Actual Positives]. Also known as 'TPR' or 'Sensitivity'
# F1 Score: 2*Precision*Recall/(Precision + Recall)
# Precision, Recall, F1 Score interpretation: Higher the better

# Precision 
# Intuitive understanding: How many of our "predicted" defaulters are "actually" defaulters

# Recall (Also known as 'TPR' or 'Sensitivity')
# Intuitive understanding: How many of the "actual defaulters", were we able to "predict correctly" as defaulters


# F1 Score: 2*Precision*Recall/(Precision + Recall) # Harmonic mean of Precision & Recall
# F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0. 
# For F1 score to be high, both precision and recall should be high
# Example: Precision = 0.9 & Recall = 0.8, then F1 score = 0.847 (F1 score is high when both Precision & Recall are high)
# Example: Precision = 0.9 & Recall = 0.2, then F1 score = 0.327 (F1 score is low when either or both Precision/Recall is/are low)



########################
# AUC and ROC Curve
########################

from sklearn.metrics import roc_curve, auc
# Predict on train data
Train_Prob = M11.predict(trainX)

# Calculate FPR, TPR and Cutoff Thresholds
fpr, tpr, cutoff = roc_curve(trainY, Train_Prob)


# Cutoff Table Creation
Cutoff_Table = pd.DataFrame()
Cutoff_Table['FPR'] = fpr 
Cutoff_Table['TPR'] = tpr
Cutoff_Table['Cutoff'] = cutoff

# Plot ROC Curve
import seaborn as sns
sns.lineplot(Cutoff_Table['FPR'], Cutoff_Table['TPR'])

# Area under curve (AUC)
auc(fpr, tpr)



############################
# Improve Model Output Using New Cutoff Point
############################

import numpy as np
Cutoff_Table['Distance'] = np.sqrt((1-Cutoff_Table['TPR'])**2 + (0-Cutoff_Table['FPR'])**2) # Euclidean Distance
Cutoff_Table['DiffBetweenTPRFPR'] = Cutoff_Table['TPR'] - Cutoff_Table['FPR'] # Max Diff. Bet. TPR & FPR

# New Cutoff Point Performance (Obtained after studying ROC Curve and Cutoff Table)
cutoffPoint = 0.202342 # Max Difference between TPR & FPR

# Classify the test predictions into classes of 0s and 1s
testX['Test_Class2'] = np.where(testX['Test_Prob'] >= cutoffPoint, 1, 0)

# Confusion Matrix
Confusion_Mat2 = pd.crosstab(testX['Test_Class2'], testY) # R, C format
Confusion_Mat2

# Model Evaluation Metrics
print(classification_report(testY, testX['Test_Class2']))
# Recall of 1s has almost doubled from 0.34 to 0.59







