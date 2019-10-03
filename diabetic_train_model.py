#!/usr/bin/env python
# coding: utf-8

# In[63]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
import joblib


# In[64]:


col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']


# In[65]:


#importing file
pima= pd.read_csv('india_pima_diabetics.csv', header=None, names = col_names)


# In[66]:


pima.head()


# In[67]:


pima.describe()


# In[68]:


pima.corr()


# In[69]:


import seaborn as sb
sb.boxplot(pima)


# In[70]:


corr = pima.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, annot=True, cmap="Reds")
plt.title('heatmap', fontsize=20)


# In[71]:


#print("Diabetes data set dimensions : {}".format(pima.shape))


# In[84]:


pima.groupby('label').size()


# In[86]:


pima.groupby('label').hist(figsize=(9, 9))


# In[72]:


col_names=['pregnant','glucose','bp','skin','insulin','bmi','pedigree','age','label']


# In[73]:


#split the dataset in features and target variable
feature_cols=['pregnant','glucose','bp','insulin','bmi','pedigree','age']
X=pima[feature_cols]
y=pima.label

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[74]:


#import the class
from sklearn.linear_model import LogisticRegression
#instantiate the model (using the default parameters)
logreg=LogisticRegression()


# In[75]:


#fit the model with data
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)


# In[76]:


#import the metrics class
from sklearn import metrics
cnf_matrix=metrics.confusion_matrix(y_test,y_pred)
#print(cnf_matrix) #26 and 11 are incorrect predictions


# In[77]:


#plotting


class_names=[0,1]
fig, ax=plt.subplots()
tick_marks=np.arange(len(class_names))
ax.xaxis.set_label_position("top")
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
plt.tight_layout()
plt.title('Confusion matrix',y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[78]:


#create heatmap 
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu",fmt='g')


# In[80]:


#Accuracy
Accuracy = metrics.accuracy_score(y_test,y_pred)
#print("Accuracy:",metrics.accuracy_score(y_test,y_pred))
#print("Precision:",metrics.precision_score(y_test,y_pred))
#print("Recall:",metrics.recall_score(y_test,y_pred))


# In[81]:


y_pred_proba=logreg.predict_proba(X_test)[::,1]
#print(y_pred_proba)
fpr,tpr,_=metrics.roc_curve(y_test, y_pred_proba)

auc=metrics.roc_auc_score(y_test,y_pred_proba)

plt.plot(fpr,tpr,label="data 1,auc="+str(auc))
plt.legend(loc=1)
plt.show


# In[57]:


#2	197	70	45	543	30.5	0.158	53
# In[59]:


y_pred_proba=logreg.predict_proba(X_test)[::,1]
#print(y_pred_proba)
fpr,tpr,_=metrics.roc_curve(y_test, y_pred_proba)

auc=metrics.roc_auc_score(y_test,y_pred_proba)

plt.plot(fpr,tpr,label="data 1,auc="+str(auc))
plt.legend(loc=1)
plt.show


# Save the model as a pickle in a file 
joblib.dump(logreg, 'dibetic.pkl')
#joblib.dump([Accuracy,y_pred], 'objects.pkl')





# In[ ]:



