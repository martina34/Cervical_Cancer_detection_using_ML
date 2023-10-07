#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Set-up libraries needed
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import joblib
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


#Loading the data  
#dataset = pd.read_csv("D:\Martina\Faculty\Graduation Project\Dataset for binary classification\database_results.csv")

data=pd.read_csv('D:/Martina/Faculty/Graduation Project/abnormal/new_database_results_abnormal.csv')


# In[3]:


data.head()


# In[4]:


#X Data
X = data.drop('ID', axis=1)


# In[5]:


#y Data
y = data['Class']
data.head()


# In[6]:


from sklearn.model_selection import train_test_split 

dataset = pd.DataFrame(data)
dataset.columns



# In[7]:


#X Data
X = data.drop('ID', axis=1)


# In[8]:


#y Data
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)
# split into test and train dataset, and use random_state=48
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)



# In[9]:


# importing SVM module
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
# kernel to be set radial bf 
classifier1 = SVC(kernel='linear')
# traininf the model
classifier1.fit(X_train,y_train)
# testing the model
y_pred = classifier1.predict(X_test)
# importing accuracy score
from sklearn.metrics import accuracy_score
#printing the accuracy of the model
#print(accuracy_score(y_test, y_pred))
SVM_Accuracy=round(accuracy_score(y_test, y_pred)*100 ,2)
print('SVM Accuracy: {:.2f}%'.format(accuracy_score(y_test, y_pred)*100))


# In[10]:


SVM_Accuracy


# In[11]:


import joblib

filename = "binary.joblib"

with open(filename, "wb") as f:
  joblib.dump(classifier1, f)


# In[12]:


#knn
missing = dataset["Kerne_A"].isna()
dataset[missing]


# In[13]:


X = dataset[-missing][[ "Kerne_Ycol"]]
y = dataset[-missing]["K/C"].values
lin_reg = sm.OLS(y, sm.add_constant(X)).fit()
print("Adjusted R-squared: {:.3f}%".format(100*lin_reg.rsquared_adj))
beta = lin_reg.params.values 
print("Estimate:", beta)


# In[14]:


dataset["Kerne_Ycol"] = dataset.apply(
    lambda row: (row.K/C - beta[0] - beta[1]*row.KerneShort)/beta[2] if np.isnan(row.Kerne_Ycol) else row.Kerne_Ycol, axis=1
)
dataset.isna().sum()


# In[15]:


dataset[missing]


# In[16]:


dataset = dataset.drop(["K/C"], axis = 1)
dataset.head()


# In[17]:


##  KNN  ##
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[18]:


data=dataset.dropna()
X = dataset.drop('ID', axis=1)
y = dataset['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# In[19]:


k = range(1,100,2) 
testing_accuracy = []
training_accuracy = []
score = 0


# In[20]:


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


# In[21]:


best_k


# In[22]:


score


# In[23]:


dataset.info()


# In[24]:


import joblib

filename = "binary.joblib"

with open(filename, "wb") as f:
  joblib.dump(knn, f)


# In[25]:


# Accuracies
Accuracies ={"KNN" : score,
             "SVM" : SVM_Accuracy} 

best_acc = max(Accuracies.values())
label = list(Accuracies.keys())[list(Accuracies.values()).index(best_acc)]
print("The best model is {} with accuracy : {}%".format(label,round(best_acc)))
# Visualising the accuracies 
plt.figure(figsize=(8,8))
plt.bar(range(len(Accuracies.keys())), Accuracies.values())
plt.title("Models Accuracies")
plt.ylabel("Percentage")
plt.xticks(range(len(Accuracies.keys())), Accuracies.keys())
plt.show()


# In[26]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.figure(figsize =(10,5))
data.groupby(['Class'])['ID'].size().sort_values(ascending=False).plot.pie()


# In[27]:


plt.figure(figsize =(20,5))
data.groupby(['Class','ID']).size().sort_values(ascending=False).head(12).plot.bar()


# In[28]:


Accuracies ={"KNN" : score,
             "SVM" : SVM_Accuracy} 

best_acc = max(Accuracies.values())
label = list(Accuracies.keys())[list(Accuracies.values()).index(best_acc)]
print("The best model is {} with accuracy : {}%".format(label,round(best_acc)))
# Comparing all the models
models = pd.DataFrame({
    'Model': [ 'KNN', 'SVM'],
    'Score': [ score, SVM_Accuracy],})
models.sort_values(by='Score', ascending=False)




# In[ ]:





# In[ ]:




