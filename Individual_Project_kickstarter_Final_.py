#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime, date
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import warnings

from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


# In[2]:


#df = pd.read_excel(filename)

df = pd.read_excel("Kickstarter.xlsx")


# In[3]:


df.head(7)


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df.info()


# In[7]:


# count the rate of successful and failed here

df.state.value_counts()


# In[8]:


# only interested in success or failure

df1 = df[(df["state"] == "successful") | (df["state"] == "failed")]


# In[9]:


df1.state.value_counts().plot.bar()


# In[10]:


df1.head(5)


# In[11]:


# converting datetime objects for deadline, state_changed_at, created_at, launched_at

df1["deadline"]= pd.to_datetime(df1["deadline"])

df1["state_changed_at"]= pd.to_datetime(df1["state_changed_at"])

df1["created_at"]= pd.to_datetime(df1["created_at"])

df1["launched_at"]= pd.to_datetime(df1["launched_at"])


df1["launch_to_deadline"]= df1["deadline"]- df1["launched_at"]

df1["launch_to_deadline"]= df1["launch_to_deadline"].dt.round('d').dt.days


df1["creation_to_launch"]= df1["launched_at"] - df1["created_at"]
df1["creation_to_launch"] = df1["creation_to_launch"].dt.round('d').dt.days


df1["launch_year"] = df1["launched_at"].dt.year
df1["launch_month"] = df1["launched_at"].dt.month


df1["deadline_year"]= df1["launched_at"].dt.year
df1["deadline_month"]= df1["deadline"].dt.month





# In[12]:


df1.info()


# In[13]:


df1.shape


# In[14]:


df1.isnull().sum()

# category and launch_to_state_change_days contains na values but i noticed removing launch_to_state_change_days also removed failure observations from state column.


# In[15]:


# drop na values from category column
df1.dropna(subset=["category"],inplace=True)


# In[16]:


df1.isnull().sum()


# In[17]:


# check duplicates

x= len(df1[df1.duplicated(subset="project_id")])
print(f'there are {x} duplicated instances in the dataset')




# In[18]:


df1.state.value_counts()


# ## Exploratory Data Analysis

# In[19]:


# converting the goal amount in USD amount to normailize

df1["usd_goal"]= round(df1["goal"]* df1["static_usd_rate"], 2)


# In[20]:


# comparing creation and launch dates

# the plot shows most projects get created and get launched right way
bins= [_*5 for _ in range(0,20)]
sns.histplot(df1.creation_to_launch, bins= bins)


# In[21]:


# examining campaign launch dates by year

plt.title("Number of campaign launches by year")
df1.launched_at_yr.value_counts(sort=True, ascending=True).plot(kind="bar",rot=0);
plt.xlabel('year');
plt.ylabel('frequency');


# In[22]:


# examining campaign launch dates by month

plt.title("Number of campaign launches by month")
df1.launched_at_month.value_counts(sort=True, ascending=True).plot(kind="bar",rot=0);
plt.xlabel('month');
plt.ylabel('frequency');


# In[23]:


# examining campaign by deadline month
# month 8 and 12 had the highest number of deadline date, the campaign launched previous month and it make sense most projects are within one month

plt.title("Number of campaign deadline by month")
df1.deadline_month.value_counts(sort=True, ascending=True).plot(kind="bar",rot=0);
plt.xlabel('month');
plt.ylabel('frequency');


# In[24]:


target = df1.state.value_counts(normalize=True) 
print(target)
plt.figure(figsize=(10,5))
sns.barplot(target.index, target.values)
plt.title('Kickstarter Success Ratio')
plt.ylabel('Percentage of Campaign', fontsize=12);


# In[25]:


country_list = df1.country.value_counts()
plt.figure(figsize=(16,5))
sns.barplot(country_list.index, country_list.values, alpha=0.8, color= "green")
plt.title('Kickstarter Countries')
plt.ylabel('Number of Campaigns', fontsize=12);


# # Feature Engineering

# In[26]:


# dummify state
df1['state'] = df1.state.astype(str)
df1['success'] = np.where(df1.state == "successful", 1, 0)
df1.success.value_counts()


# In[27]:



# which countries has the highest success rate
country_success = df1[df1.success == 1].groupby(['country']).size()
print(country_success)

plt.figure(figsize=(16,5))
sns.barplot(x=country_success.index, y=country_success.values, color="green")
plt.xticks(rotation=45);


# In[28]:


# which countries has the lowest success rate
country_fail = df1[df1.success == 0].groupby(['country']).size()
print(country_fail)

plt.figure(figsize=(16,5))
sns.barplot(x=country_fail.index, y=country_fail.values, color="red")
plt.xticks(rotation=45);


# In[29]:


country_dict = {'AT':'OTHER', 'AU': 'OTHER', 'BE':'OTHER', 'CA': 'OTHER', 'CH':'OTHER', 'DE':'OTHER','DK':'OTHER', 'ES': 'OTHER', 'FR': 'OTHER',
 'GB': 'OTHER', 'HK': 'OTHER', 'IE':'OTHER','IT': 'OTHER','LU':'OTHER', 'MX': 'OTHER', 'NL': 'OTHER','NO':'OTHER','NZ':'OTHER', 'SE': 'OTHER','SG':'OTHER'}
df1 = df1.replace({"country": country_dict})


# In[30]:


country_success = df1[df1.success == 1].groupby(['country']).size()
print(country_success)

plt.figure(figsize=(16,5))
sns.barplot(x=country_success.index, y=country_success.values)
plt.xticks(rotation=45);


# In[31]:


# Make dummies
df1 = pd.get_dummies(columns=['country'], drop_first=True, data=df1)
df1.columns


# In[32]:


df1.info()


# In[33]:


# main category + success
category_success = df1[df1.success == 1].groupby(['category']).size()
print(category_success)
plt.figure(figsize=(16,5))
sns.barplot(x=category_success.index, y=category_success.values)
plt.xticks(rotation=45);


# In[34]:


# Make category dummies
df1 = pd.get_dummies(columns=['category'], drop_first=True, data=df1)
df1.columns


# In[35]:


# campaign duration, preparation duration
from datetime import datetime, date, timedelta

prep_date = df1.created_at.values
start_date = df1.launched_at.values
end_date = df1.deadline.values

df1['campaign_duration'] = pd.to_timedelta(end_date - start_date).days
df1['preparation_duration'] = pd.to_timedelta(start_date - prep_date).days


# In[36]:


df1.head(4)


# In[37]:


# spotlight and staff_pick
df1 = pd.get_dummies(columns = ['spotlight','staff_pick'], drop_first=True, data=df1)
df1.head(5)


# In[38]:


ks= df1[["usd_goal","success","country_US","campaign_duration","preparation_duration","category_Apps",
         "category_Blues","category_Comedy","category_Experimental","category_Festivals","category_Flight","category_Gadgets",
         "category_Hardware","category_Immersive","category_Makerspaces","category_Musical",
         "category_Places","category_Plays","category_Robots","category_Shorts","category_Software",
         "category_Sound","category_Spaces","category_Thrillers","category_Wearables",
         "category_Web","category_Webseries"]]

ks.head()


# In[39]:


ks.shape


# In[40]:


ks.info()


# ## Build The Classification Model

# In[41]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score, recall_score, precision_recall_curve,f1_score, fbeta_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import roc_curve, auc


# In[42]:


# baseline
subset_1= ks.loc[:,["success","usd_goal","country_US","campaign_duration"]]

subset_1.head()


# # Logistic regression Model

# In[43]:


# define x and y

X, y = subset_1.drop(['success'], axis=1), subset_1.success

# split the data and standarize

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)


# In[44]:


# build logistic regression model and predict

lr = LogisticRegression()
lr.fit(X_train_sc, y_train)
lr_pred = lr.predict(X_test_sc)


# In[45]:


# evaluate the model
lr_ac = lr.score(X_test_sc, y_test)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)

lr_y_score = lr.predict_proba(X_test_sc)[:,1]
lr_fpr, lr_tpr, lr_auc_thresholds = roc_curve(y_test, lr_y_score)
roc_auc_lr = auc(lr_fpr, lr_tpr)


print(classification_report(y_test, lr_pred))


# In[46]:


print('Logistic Regression validation metrics with subset 1: \n Accuracy: %.4f \n Precision: %.4f \n Recall: %.4f \n F1: %.4f \n ROC: %.4f' %
        (lr_ac, 
         lr_precision, 
         lr_recall,
         lr_f1,
         roc_auc_lr)
     )


# In[47]:


# baseline 2
subset_2= ks.loc[:,["success","usd_goal","country_US","campaign_duration","preparation_duration"]]

subset_2.head()


# In[48]:


# define x and y

x2, y2 = subset_2.drop(['success'], axis=1), subset_2.success

# split the data and standarize

X_train, X_test, y_train, y_test = train_test_split(x2, y2, test_size=0.2, random_state=5)

scaler = StandardScaler()
X_train_sc2 = scaler.fit_transform(X_train)
X_test_sc2 = scaler.transform(X_test)


# In[49]:


# build the model and predict
lr2 = LogisticRegression(C=1000) #small regularization applied 
lr2.fit(X_train_sc2, y_train)
lr2_pred = lr2.predict(X_test_sc2)


# In[50]:


# evaluate the model

lr2_confusion = confusion_matrix(y_test, lr2_pred)

lr2_ac = lr2.score(X_test_sc2, y_test)
lr2_precision = precision_score(y_test, lr2_pred)
lr2_recall = recall_score(y_test, lr2_pred)
lr2_f1 = f1_score(y_test, lr2_pred)

lr2_y_score = lr2.predict_proba(X_test_sc2)[:,1]
lr2_fpr, lr2_tpr, lr2_auc_thresholds = roc_curve(y_test, lr2_y_score)
roc_auc_lr2 = auc(lr2_fpr, lr2_tpr)

print(classification_report(y_test, lr2_pred))


# In[51]:


print('Logistic Regression validation metrics with subset 2: \n Accuracy: %.4f \n Precision: %.4f \n Recall: %.4f \n F1: %.4f \n ROC: %.4f' %
        (lr2_ac, 
         lr2_precision, 
         lr2_recall,
         lr2_f1,
         roc_auc_lr2)
     )


# In[52]:


# all features (ks)
x3, y3 = ks.drop(['success'], axis=1), ks.success

X_train, X_test, y_train, y_test = train_test_split(x3, y3, test_size=0.2, random_state=5)

scaler = StandardScaler()
X_train_sc3 = scaler.fit_transform(X_train)
X_test_sc3 = scaler.transform(X_test)


# In[53]:


lr3 = LogisticRegression(C=0.001) # with regularization
lr3.fit(X_train_sc3, y_train)
lr3_pred = lr3.predict(X_test_sc3)
lr3_confusion = confusion_matrix(y_test, lr3_pred)

lr3_ac = lr3.score(X_test_sc3, y_test)
lr3_precision = precision_score(y_test, lr3_pred)
lr3_recall = recall_score(y_test, lr3_pred)
lr3_f1 = f1_score(y_test, lr3_pred)

lr3_y_score = lr3.predict_proba(X_test_sc3)[:,1]
lr3_fpr, lr3_tpr, lr3_auc_thresholds = roc_curve(y_test, lr3_y_score)
roc_auc_lr3 = auc(lr3_fpr, lr3_tpr)

print(classification_report(y_test, lr3_pred))


# In[54]:


# accuracy has improved greatly but the f1 score not satisfactory 
print('Logistic Regression validation metrics with full dataset: \n Accuracy: %.4f \n Precision: %.4f \n Recall: %.4f \n F1: %.4f \n ROC: %.4f' %
        (lr3_ac, 
         lr3_precision, 
         lr3_recall,
         lr3_f1,
         roc_auc_lr3)
     )


# In[55]:


plt.figure(figsize=(5,5))
lw = 2
plt.plot(lr_fpr, lr_tpr, color='black', lw=lw, 
         label='subset 1 (area= %0.2f)' %roc_auc_lr)
plt.plot(lr2_fpr, lr2_tpr, color='blue', lw=lw, 
         label='subset 2 (area= %0.2f)' %roc_auc_lr2)
plt.plot(lr3_fpr, lr3_tpr, color='green', lw=lw, 
         label='full data (area= %0.2f)' %roc_auc_lr3)
plt.plot([0,1],[0,1],c='violet',ls='--')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])


plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right');


# In[ ]:





# # hyperparameter tuning logistic regression
# 
# 
# 

# In[56]:


from sklearn.model_selection import GridSearchCV


# In[57]:


# define x and y
X4, Y4 = ks.drop(['success'], axis=1), ks.success
# split into train and test set
X_mid, X_test, y_mid, y_test = train_test_split(X4, Y4, test_size=0.2, random_state=5)
X_train, X_val, y_train, y_val = train_test_split(X_mid, y_mid, test_size=0.2, random_state=5)


# In[58]:


# standarize the x values
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc = scaler.transform(X_val)


# In[59]:


# build a model
lr = LogisticRegression()
lr.fit(X_train_sc, y_train)
lr_pred = lr.predict(X_val_sc)
lr_confusion = confusion_matrix(y_val, lr_pred)

lr_ac = lr.score(X_val_sc, y_val)
lr_precision = precision_score(y_val, lr_pred)
lr_recall = recall_score(y_val, lr_pred)
lr_f1 = f1_score(y_val, lr_pred)

print(classification_report(y_val, lr_pred))


# In[60]:


# logistic regression grid search to improve precision, and F1 score
penalty = ['l1', 'l2']
C = [0.001,0.01,0.1,1,10,100,1000]
param_grid = dict(C=C, penalty=penalty)
lr_grid_search = LogisticRegression()
lr_grid = GridSearchCV(lr_grid_search, param_grid, cv=5, scoring='f1', verbose=2, n_jobs=-1, refit = True)
lr_grid.fit(X_train_sc, y_train)
lr_grid_preds = lr_grid.predict(X_val_sc)

lr_grid_best_params = lr_grid.best_params_
lr_grid_best_estimator = lr_grid.best_estimator_
lr_grid_best_cm = confusion_matrix(y_val,lr_grid_preds)
lr_grid_best_cr = classification_report(y_val,lr_grid_preds)
print(lr_grid_best_params, lr_grid_best_estimator, lr_grid_best_cm, lr_grid_best_cr)


# In[61]:


# better result seen with hyperparameter tuning for f1 score
print( "logistic regression with hyperparameter: \n Accuracy: %.4f \n Precision: %.4f \n Recall: %.4f \n F1: %.4f" %
        (lr_ac, 
         lr_precision, 
         lr_recall,
         lr_f1)
     )


# In[62]:


### extra classification analysis was done using random forest


# ## random forest

# In[63]:


from sklearn.ensemble import RandomForestClassifier


# In[64]:


# define x and y
X5, Y5 = ks.drop(['success'], axis=1), ks.success
# split into train/test set
X_train, X_test, y_train, y_test = train_test_split(X5, Y5, test_size=0.2, random_state=5)
# standarize
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)


# In[65]:


#Build Random forest model
rf = RandomForestClassifier()
rf.fit(X_train_sc, y_train)
rf_pred = rf.predict(X_test_sc)
rf_confusion = confusion_matrix(y_test, rf_pred)

rf_ac = rf.score(X_test_sc, y_test)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print(classification_report(y_test, rf_pred))


# In[66]:


print('Random Forest validation metrics: \n Accuracy: %.4f \n Precision: %.4f \n Recall: %.4f \n F1: %.4f' %
        (rf_ac, 
         rf_precision, 
         rf_recall,
         rf_f1)
     )


# In[ ]:





# ## Clustering model

# In[68]:


fig_dims = (12, 12)
fig, ax = plt.subplots(figsize=fig_dims)
sns.heatmap(ks.corr());


# # PCA analysis

# In[69]:


X_pca= ks.drop(['success'], axis=1)
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

X_pca_scaler= scale(X_pca)


# In[70]:


from sklearn.decomposition import PCA
pca = PCA().fit(X_pca_scaler)

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,6)

fig, ax = plt.subplots()
xi = np.arange(1, 27, step=1)
y = np.cumsum(pca.explained_variance_ratio_)
plt.ylim(0.0,1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')

plt.xlabel('Number of Components')
plt.xticks(np.arange(0, 30, step=1)) #change from 0-based array index to 1-based human-readable label
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')

plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

ax.grid(axis='x')
plt.show()


# In[71]:


# elbow method to determine the optimal k value


# In[72]:


# define x and remove the target variable
X= ks.drop(['success'], axis=1)


# In[73]:


from sklearn.cluster import KMeans
wss=[]# within clusters sum of errors
for i in range(2,8):
  kmeans= KMeans(n_clusters=i)
  model= kmeans.fit(X)
  wss.append(model.inertia_)

from matplotlib import pyplot
pyplot.plot([2,3,4,5,6,7],wss)
pyplot.title("Elbow Method")
pyplot.ylabel("WSS")
pyplot.xlabel("Number of clusters")


# from this graph, the optimal cluster should be 4

# In[74]:


## lets use data from the orginal data


# In[75]:


# examine the categories
X_cluster2= df1[["pledged","backers_count"]]

# standarize the x value
X_cluster2sc= scaler.fit_transform(X_cluster2)


# In[76]:


# k-mean model

kmeans= KMeans(n_clusters=4, random_state=0)
kmeans.fit(X_cluster2sc)


# In[77]:


kmeans= KMeans(n_clusters=4)
model= kmeans.fit(X_cluster2sc)
labels= model.predict(X_cluster2sc)


# In[78]:


# plot the graph
from matplotlib import pyplot
pyplot.scatter(df1.pledged,df1.backers_count, c=labels, cmap= "rainbow")

pyplot.xlabel("pledged")
pyplot.ylabel("backers_count")



# In[ ]:





# ![Screen Shot 2021-12-07 at 10.01.56 PM.png](data:image/png;base64,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QAABBBBAAAEEkk6AICvpmowKI4AAAggggAACCCCAQCILEGQlcutQNwQQQAABBBBAAAEEEEg6AYKspGsyKowAAggggAACCCCAAAKJLECQlcitQ90QQAABBBBAAAEEEEAg6QQIspKuyagwAggggAACCCCAAAIIJLIAQVYitw51QwABBBBAAAEEEEAAgaQTIMhKuiajwggggAACCCCAAAIIIJDIAgRZidw61A0BBBBAAAEEEEAAAQSSToAgK+majAojgAACCCCAAAIIIIBAIgsQZCVy61A3BBBAAAEEEEAAAQQQSDoBgqykazIqjAACCCCAAAIIIIAAAoksQJCVyK1D3RBAAAEEEEAAAQQQQCDpBAiykq7JqDACCCCAAAIIIIAAAggksgBBViK3DnVDAAEEEEAAAQQQQACBpBMgyEq6JqPCCCCAAAIIIIAAAgggkMgCVRKlcq+++qqNHj3a0tLSrHXr1nbdddfZ8uXL7Z577rG1a9faTjvtZDfccINlZmba5s2b7b777rNZs2ZZdna23XTTTda0aVN/K88//7x99NFHlp6ebhdddJH16NHDbx83bpz973//s23bttmhhx5qp556aqLcOvVAAAEEEEAAAQQQQACBSiSQED1ZS5cutXfffdcGDRpkQ4cO9YHQmDFj7IknnrCjjz7annzySatVq5aNHDnS0+tZ77Vdnz/++ON++6+//mpffPGFP8edd95pQ4YM8edSYDVs2DC74/Y7/PPYsWNN+1IQQAABBBBAAAEEEEAAgfIWSIggSzelQCgnJ8e2bt3qnxs0aGBTp061gw46yN/zYYcdZuqNUhk/frzpvYo+1355eXk+wOrTp49VrVrVmjVr5nu3fvzxR9NDPV3NWzT3PWHaR8EYBQEEEEAAAQQQQAABBBAob4GEGC7YqFEjO+aYY+ycc87xQVDXrl2tQ4cOVqNGDcvIyPD3rH1WrFjhX+tZ71X0ufZbs2aNH17YsWNHv13/1K9f35YtW+bfK2gLio796aefgrc8I4AAAggggAACCCCAAALlJpAQQZbmXKmXSsMDNcdKQ/3UWxWPomGLo0aN8pd+5JFHrEmTJjGvhuadxeO6Mb/RBL8g7ZAYDUQ70A6JIRD/WvDfQvzbQDWgHWiHxBCgFokukBBB1oQJE6xx48ZWt25d79WrVy/74YcfbMOGDX74oHqrNG9LPVMq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)

# In[ ]:





# In[ ]:





# In[ ]:




