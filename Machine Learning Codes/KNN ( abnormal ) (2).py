#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing dependencies
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Set-up libraries needed
# data analysis
import pandas as pd
import numpy as np

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


#Loading the data  
#dataset = pd.read_csv("D:\Martina\Faculty\Graduation Project\Dataset for binary classification\database_results.csv")

dataset=pd.read_csv('D:/Martina/Faculty/Graduation Project/abnormal/new_database_results_abnormal.csv')


# In[3]:


missing = dataset["Kerne_A"].isna()
dataset[missing]


# In[4]:


X = dataset[-missing][[ "Kerne_Ycol"]]
y = dataset[-missing]["K/C"].values
lin_reg = sm.OLS(y, sm.add_constant(X)).fit()
print("Adjusted R-squared: {:.3f}%".format(100*lin_reg.rsquared_adj))
beta = lin_reg.params.values 
print("Estimate:", beta)


# In[5]:


dataset["Kerne_Ycol"] = dataset.apply(
    lambda row: (row.K/C - beta[0] - beta[1]*row.KerneShort)/beta[2] if np.isnan(row.Kerne_Ycol) else row.Kerne_Ycol, axis=1
)
dataset.isna().sum()


# In[6]:


dataset[missing]


# In[7]:


dataset = dataset.drop(["K/C"], axis = 1)
dataset.head()


# In[8]:


##  KNN  ##
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[9]:


data=dataset.dropna()
X = dataset.drop('ID', axis=1)
y = dataset['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# In[10]:


k = range(1,100,2) 
testing_accuracy = []
training_accuracy = []
score = 0


# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

score=0
for i in k: 
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train) 
    y_predict_train = knn.predict(X_train)
    training_accuracy.append(accuracy_score(y_train, y_predict_train)) 
    y_predict_test = knn.predict(X_test)
    acc_score  = accuracy_score(y_test,y_predict_test)
    testing_accuracy.append( acc_score  ) 
    if score <  acc_score  : 
        score =  acc_score  
    best_k = i
print('This is the best K for KNeighbors Classifier: ', best_k, '\nAccuracy score is: ', score)


# In[12]:


dataset.info()


# In[13]:


import joblib

filename = "binary.joblib"

with open(filename, "wb") as f:
  joblib.dump(knn, f)


# In[ ]:





# In[ ]:




