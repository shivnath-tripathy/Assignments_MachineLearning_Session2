
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_boston
boston_dataset = load_boston()
boston_data = pd.DataFrame(boston_dataset.data)


# In[4]:


boston_data.head()  # Data Structure for boston


# In[9]:


print(boston_dataset.DESCR)


# In[7]:


print(boston_dataset.feature_names)


# In[8]:


boston_data.columns = boston_dataset.feature_names


# In[10]:


boston_data.head()


# In[11]:


boston_data['PRICE']=boston_dataset.target


# In[13]:


boston_data.head()


# In[14]:


from sklearn.linear_model import  LinearRegression


# In[15]:


X = boston_data.drop('PRICE', axis=1) # add the feature data to a variable
Y = boston_data['PRICE'] # add the target or label data to a variable
lm = LinearRegression() # create a linear regression object
lm.fit(X,Y) # try to fit the X and Y value to get w(weights or coefficiet) and b value


# In[16]:


print("Estimated Intercept Coefficient i.e b =",lm.intercept_)
print("Estimated Coefficient (Weights for each feature):",lm.coef_)


# In[17]:


print("Coefficient in Table Structure for each feature")
EstCoef = pd.DataFrame(list(zip(X.columns,lm.coef_)),columns = ['Features','Coefficient(Weights)'])
print("Highest Positive coefficient has biggest impact on the target!")
EstCoef


# In[18]:


# Split data for Training and testing 
from sklearn.model_selection import train_test_split


# In[19]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state=5)


# In[20]:


print("Shape of X Train Data",X_train.shape)
print("Shape of X Test Data",X_test.shape)
print("Shape of Y Train Original Output",Y_train.shape)
print("Shape of Y Test Original Output",Y_test.shape)


# In[21]:


# Fit and find the weights and b value using Linear regression
lm = LinearRegression()
lm.fit(X_train,Y_train)


# In[22]:


pred_train = lm.predict(X_train) # Predicting train data itself
pred_test = lm.predict(X_test) # Perdicting test data


# In[23]:


#Display Mean Square Error for Perdicted data for training set vs Original Output
print ('Fit a model X_train, and calculate MSE with Y_train:', np.mean((Y_train-pred_train) ** 2))


# In[24]:


#Display Mean Square Error for Perdicted data for testing set vs Original Output
print ('Fit a model X_train, and calculate MSE with X_test, Y_test:', np.mean((Y_test - pred_test) ** 2))


# In[25]:


# Visualize difference between predicted and original data with a horizontal line at 0
# Plot line 0 means value predicted and original data are matching
get_ipython().magic('matplotlib inline')
plt.scatter(pred_train,pred_train- Y_train,c="b",s=40,alpha=0.9, label = 'Train data')
plt.scatter(pred_test,pred_test- Y_test,c="r",s=40,alpha=0.9,  label = 'Test data')
## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)
 
## plotting legend
plt.legend(loc = 'upper right')

# Y label
plt.ylabel("Residuals")

## plot title
plt.title("Residual plot using training and test data")

plt.show()

