#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Set-up libraries needed
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#import joblib
import pandas as pd
import numpy as np
import os

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from IPython.display import Image

# preprocessing (pre-modeling)
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# modeling
import statsmodels.api as sm
from sklearn.neighbors import KNeighborsClassifier


# evaluation (post-modeling)
from sklearn.metrics import *
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from scipy.special import logit


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Importing Data file
data = pd.read_csv("D:/Martina/Faculty/Graduation Project/Dataset for binary classification/database_results.csv")
dataset = pd.DataFrame(data)
dataset.columns

#X Data
X = data.drop('ID', axis=1)


#y Data
y = data['Class']


# In[3]:


dataset.info()


# In[4]:


dataset.describe().transpose()



# In[5]:


#Splitting data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)

#Splitted Data
#print('X_train shape is ' , X_train.shape)
#print('X_test shape is ' , X_test.shape)
#print('y_train shape is ' , y_train.shape)
#print('y_test shape is ' , y_test.shape)


# In[22]:


from sklearn.svm import SVC

# Building a Support Vector Machine on train data
svc_model = SVC(C= .1, kernel='linear', gamma= 1)
svc_model.fit(X_train, y_train)

prediction = svc_model .predict(X_test)
# importing accuracy score
from sklearn.metrics import accuracy_score
# check the accuracy on the training set
print(svc_model.score(X_train, y_train))
SVM_Accuracy=round(svc_model.score(X_test, y_test)*100 ,2)
print('SVM Accuracy: {:.2f}%'.format(svc_model.score(X_test, y_test)*100))


# In[23]:


SVM_Accuracy


# In[10]:


print("Confusion Matrix:\n",confusion_matrix(prediction,y_test))


# In[11]:


# Building a Support Vector Machine on train data
svc_model = SVC(kernel='rbf')
svc_model.fit(X_train, y_train)


# In[12]:


print(svc_model.score(X_train, y_train))
print(svc_model.score(X_test, y_test))


# In[13]:


#Building a Support Vector Machine on train data(changing the kernel)
svc_model = SVC(kernel='poly')
svc_model.fit(X_train, y_train)

prediction = svc_model.predict(X_test)

print(svc_model.score(X_train, y_train))
print(svc_model.score(X_test, y_test))


# In[14]:


svc_model = SVC(kernel='sigmoid')
svc_model.fit(X_train, y_train)

prediction = svc_model.predict(X_test)

print(svc_model.score(X_train, y_train))
print(svc_model.score(X_test, y_test))


# In[20]:


#Applying KNeighborsClassifier Model 

'''
#sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform’, algorithm=’auto’, leaf_size=30,
#                                       p=2, metric='minkowski’, metric_params=None,n_jobs=None)
'''

KNNClassifierModel = KNeighborsClassifier(n_neighbors= 5,weights ='uniform', # it can be distance
                                          algorithm='auto') 
KNNClassifierModel.fit(X_train, y_train)

#Calculating Details
print('KNNClassifierModel Train Score is : ' , KNNClassifierModel.score(X_train, y_train))
print('KNNClassifierModel Test Score is : ' , KNNClassifierModel.score(X_test, y_test))
print('----------------------------------------------------')
#Calculating Details
print('KNNClassifierModel Train Score is : ' , KNNClassifierModel.score(X_train, y_train))
KNN_Accuracy=round(KNNClassifierModel.score(X_test, y_test)*100 ,2)
print('KNNClassifierModel Test Score is : ' , KNNClassifierModel.score(X_test, y_test))
print('----------------------------------------------------')



# In[21]:


KNN_Accuracy


# In[27]:


#Calculating Prediction
y_pred = KNNClassifierModel.predict(X_test)
y_pred_prob = KNNClassifierModel.predict_proba(X_test)
print('Predicted Value for KNNClassifierModel is : ' , y_pred[:10])
print('Prediction Probabilities Value for KNNClassifierModel is : ' , y_pred_prob[:10])


# In[28]:


#import k-folder
from sklearn.model_selection import cross_val_score
# use the same model as before
knn = KNeighborsClassifier(n_neighbors = 5)
# X,y will automatically devided by 5 folder, the scoring I will still use the accuracy
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
# print all 5 times scores 
print(scores)
# [ 0.96666667  1.          0.93333333  0.96666667  1.        ]
# then I will do the average about these five scores to get more accuracy score.
print(scores.mean())


# In[14]:


#Calculating Confusion Matrix
CM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix is : \n', CM)

# drawing confusion matrix
sns.heatmap(CM, center = True)
plt.show()


# In[15]:


data.hist(figsize=(20,20))
plt.show()


# In[24]:


# Accuracies
Accuracies ={"KNN" : KNN_Accuracy,
             "SVM" : SVM_Accuracy} 

best_acc = max(Accuracies.values())
label = list(Accuracies.keys())[list(Accuracies.values()).index(best_acc)]
print("The best model is {} with accuracy : {}%".format(label,round(best_acc)))
# Visualising the accuracies 
plt.figure(figsize=(4,4))
plt.bar(range(len(Accuracies.keys())), Accuracies.values())
plt.title("Models Accuracies")
plt.ylabel("Percentage")
plt.xticks(range(len(Accuracies.keys())), Accuracies.keys())
plt.show()


# In[25]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.figure(figsize =(10,5))
data.groupby(['Class'])['ID'].size().sort_values(ascending=False).plot.pie()


# In[29]:


import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
# choose k between 1 to 31
k_range = range(1, 31)
k_scores = []
# use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())
# plot to see clearly
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


# In[30]:


plt.figure(figsize =(20,5))
data.groupby(['Class','ID']).size().sort_values(ascending=False).head(12).plot.bar()


# In[31]:


Accuracies ={"KNN" : KNN_Accuracy,
             "SVM" : SVM_Accuracy} 

best_acc = max(Accuracies.values())
label = list(Accuracies.keys())[list(Accuracies.values()).index(best_acc)]
print("The best model is {} with accuracy : {}%".format(label,round(best_acc)))
# Comparing all the models
models = pd.DataFrame({
    'Model': [ 'KNN', 'SVM'],
    'Score': [ KNN_Accuracy, SVM_Accuracy],})
models.sort_values(by='Score', ascending=False)


# In[32]:


import joblib

filename = "binary.joblib"

with open(filename, "wb") as f:
  joblib.dump(svc_model, f)


# In[33]:


import joblib

filename = "binary2.joblib"

with open(filename, "wb") as f:
  joblib.dump(KNNClassifierModel, f)


# In[ ]:




