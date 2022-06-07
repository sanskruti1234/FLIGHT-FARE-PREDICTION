#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv(r'C:\Users\sameer narwade\Downloads\Data_Train.xlsx - Sheet1.csv')
df.head()


# In[3]:


df.nunique()


# In[4]:


df.isnull().sum()


# In[5]:


df.dropna(inplace=True)


# In[6]:


df.dtypes


# In[7]:


def convert_datetime(col):
    df[col]=pd.to_datetime(df[col])
    


# In[8]:


convert_datetime('Date_of_Journey')


# In[9]:


df.columns


# In[10]:


lists = ['Date_of_Journey','Dep_Time','Arrival_Time']
for i in lists:
    convert_datetime(i)


# In[11]:


df['journey_day']=df['Date_of_Journey'].dt.day
df['journey_month']=df['Date_of_Journey'].dt.month


# In[12]:


df.head()


# In[13]:


def hours_min(df,col):
    df[col + '_hour'] = df[col].dt.hour
    df[col + '_minute'] = df[col].dt.minute


# In[14]:


hours_min(df,'Dep_Time')
hours_min(df,'Arrival_Time')


# In[15]:


df.head()


# In[16]:


def drop_cols(col):
    df.drop(col,axis=1,inplace=True)


# In[17]:


drop_cols('Dep_Time')
drop_cols('Arrival_Time')
drop_cols('Date_of_Journey')


# In[18]:


duration = list(df['Duration'])


# In[19]:


for i in range(len(duration)):
    if len(duration[i].split(' '))==2:
        pass
    else:
        if 'h' in duration[i]:
            duration[i] = duration[i] + ' 0m'
        else:
            duration[i] = '0h ' + duration[i]


# In[20]:


df['Duration'] = duration


# In[21]:


df.head()


# In[22]:


def hour(col):
    return col.split(' ')[0][0:-1]

def minu(col):
    return col.split(' ')[1][0:-1]


# In[23]:


df['Duration_hour'] = df['Duration'].apply(hour)
df['Duration_min'] = df['Duration'].apply(minu)


# In[24]:


df.head()


# In[25]:


drop_cols('Duration')


# In[26]:


df.dtypes


# In[27]:


df['Duration_hour'] = df['Duration_hour'].astype(int)
df['Duration_min'] = df['Duration_min'].astype(int)


# In[28]:


cat_cols = [col for col in df.columns if df[col].dtype == 'O']
cat_cols


# In[29]:


num_cols = [col for col in df.columns if df[col].dtype != 'O']
num_cols


# In[30]:



def anaylsis(col):
    plt.figure(figsize=(30,10))
    sns.boxplot(df[col],'Price',data=df.sort_values('Price',ascending=False))


# In[31]:


anaylsis('Airline')


# In[32]:


anaylsis('Source')


# In[33]:


anaylsis('Destination')


# In[34]:


anaylsis('Total_Stops')


# In[35]:


def dummies(col):
    return pd.get_dummies(df[col],drop_first=True)


# In[36]:


airline = dummies('Airline')
airline


# In[37]:


source =dummies('Source')
source.head()


# In[38]:


destination = dummies('Destination')
destination.head()


# In[39]:


stops = dummies('Total_Stops')
stops.head()


# In[40]:


df['Route_1'] =df['Route'].str.split('→').str[0]
df['Route_2'] =df['Route'].str.split('→').str[1]
df['Route_3'] =df['Route'].str.split('→').str[2]
df['Route_4'] =df['Route'].str.split('→').str[3]
df['Route_5'] =df['Route'].str.split('→').str[4]


# In[41]:


df.head()


# In[42]:


df.isnull().sum()


# In[43]:


for i in ['Route_3','Route_4','Route_5']:
    df[i] = df[i].fillna('None')


# In[44]:


df['Total_Stops'].unique()


# In[45]:


dict = {'non-stop': '0', '2 stops':'2', '1 stop':'1', '3 stops':'3', '4 stops':'4'}
df['Total_Stops'] = df['Total_Stops'].map(dict)


# In[46]:


df.head()


# In[47]:


from sklearn.preprocessing import LabelEncoder


# In[48]:


le = LabelEncoder()


# In[49]:


for i in ['Route_1','Route_2','Route_3','Route_4','Route_5']:
    df[i] = le.fit_transform(df[i])


# In[50]:


df.head()


# In[51]:


drop_cols('Route')
drop_cols('Additional_Info')
drop_cols('Airline')
drop_cols('Source')
drop_cols('Destination')


# In[52]:


df.head()


# In[53]:


data_ = pd.concat([df,airline,source,destination],axis=1)


# In[54]:


data_.head()


# In[55]:


def plot(df,col):
    fig,(ax1,ax2)=plt.subplots(2,1)
    sns.distplot(df[col],ax=ax1)
    sns.boxplot(df[col],ax=ax2)


# In[56]:


plot(data_,'Price')


# In[57]:


data_['Price'] = np.where(data_['Price']>=40000,data_['Price'].median(),data_['Price'])


# In[58]:


plot(data_,'Price')


# In[59]:


X = data_.drop(['Price'],axis=1)
X.head()


# In[60]:


Y = data_['Price']
Y.head()


# In[61]:


from sklearn.feature_selection import mutual_info_classif


# In[62]:


mutual_info_classif(X,Y)


# In[63]:


imp = pd.DataFrame(mutual_info_classif(X,Y),index=X.columns)
imp


# In[64]:


imp.columns = ['importance']


# In[65]:


imp.sort_values(by='importance',ascending = False)


# In[66]:


from sklearn.model_selection import train_test_split


# In[67]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)


# In[68]:


from sklearn import metrics
import pickle


# In[69]:


def predict(ml_model,dump):
    model =ml_model.fit(X_train,Y_train)
    print('training score:{}'.format(model.score(X_train,Y_train)))
    y_predict = model.predict(X_test)
    print('predictions:\n {}'.format(y_predict))
    print('\n')
    r2_score = metrics.r2_score(Y_test,y_predict)
    print('r2_score is :{}'.format(r2_score))
    print('MAE:',metrics.mean_absolute_error(Y_test,y_predict))
    print('MSE:',metrics.mean_squared_error(Y_test,y_predict))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test,y_predict)))
    
    sns.distplot(Y_test-y_predict)
    
    if dump==1:
        with open('project_model','wb') as f:
            pickle.dump(model,f)


# In[70]:


from sklearn.ensemble import RandomForestRegressor


# In[71]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier


# In[72]:


predict(RandomForestRegressor(),1)


# In[73]:


predict(LinearRegression(),0)


# In[74]:


predict(DecisionTreeClassifier(),0)


# In[75]:


from sklearn.model_selection import RandomizedSearchCV


# In[76]:


reg_rf=RandomForestRegressor()


# In[77]:


n_estimators = [int(x) for x in np.linspace(start=100,stop=1200,num=6)]
max_depth = [int(x) for x in np.linspace(start=5,stop=30,num=4)]


# In[78]:


random_grid ={
    'n_estimators':n_estimators,
    'max_features':['auto','sqrt'],
    'max_depth':max_depth,
    'min_samples_split':[5,10,18,100,16]  
}


# In[79]:


random_grid


# In[80]:


rf_random = RandomizedSearchCV(estimator=reg_rf,param_distributions = random_grid,cv=3,verbose=2,n_jobs=-1)


# In[81]:


rf_random.fit(X_train,Y_train)


# In[82]:


rf_random.best_params_


# In[83]:


prediction=rf_random.predict(X_test)


# In[84]:


sns.distplot(Y_test-prediction)


# In[85]:


metrics.r2_score(Y_test,prediction)


# In[ ]:




