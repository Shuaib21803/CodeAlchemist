#!/usr/bin/env python
# coding: utf-8

# In[52]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd


# In[53]:


df_train = pd.read_csv("../data/train.csv")
df_test = pd.read_csv("../data/test.csv")


# In[54]:


df_train["Satisfaction in work life"] = df_train["Job Satisfaction"].combine_first(df_train['Study Satisfaction'])
df_train["Pressure in work life"] = df_train["Work Pressure"].combine_first(df_train['Academic Pressure'])
df_test["Satisfaction in work life"] = df_test["Job Satisfaction"].combine_first(df_test['Study Satisfaction'])
df_test["Pressure in work life"] = df_test["Work Pressure"].combine_first(df_test['Academic Pressure'])
df_train.drop(columns = ["Working Professional or Student","Job Satisfaction","Study Satisfaction", "Work Pressure", "Academic Pressure", "id","Name"], inplace = True)
df_test.drop(columns = ["Working Professional or Student","Job Satisfaction","Study Satisfaction", "Work Pressure", "Academic Pressure"], inplace = True)
df_train.head()


# In[55]:


df_column = df_train.columns
for column in df_column:
    if df_train[column].dtype in ["float64", "int64"]:
        mean = df_train[column].mean()
        df_train[column] = df_train[column].fillna(mean)
df_train.head()


# In[56]:


df_train.ffill(inplace = True)
df_train.head()


# In[57]:


df_train.isnull().any()


# In[58]:


df_column = df_test.columns
for column in df_column:
    if df_test[column].dtype in ["float64", "int64"]:
        mean = df_test[column].mean()
        df_test[column] = df_test[column].fillna(mean)
df_train.head()


# In[59]:


df_test.ffill(inplace = True)
df_test.head()


# In[60]:


string_columns = df_train.select_dtypes(include = ["object"]).columns
labels_encoders = {}
for column in string_columns:
    le = LabelEncoder()
    df_train[column] = le.fit_transform(df_train[column])
    labels_encoders[column] = le
df_train.head()


# In[61]:


df_test.head()


# # PyTorch MODEL 

# In[62]:


# split feature and target
# dataset class
# train and val split
# normalize
# data loader
# model


# In[63]:


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[64]:


class MentalHealthDataset(Dataset):
    def __init__(self,features,labels):
        self.features=torch.FloatTensor(np.array(features))
        self.labels=torch.LongTensor(np.array(labels))
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.features[index],self.labels[index]    


# In[65]:


class DepressionClassifier(nn.Module):
    def __init__(self, input_size):
        super(DepressionClassifier, self).__init__()
        self.layers=nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )
    def forward(self,x):
        return self.layers(x)    


# In[66]:


def prepare_data(df,target_column="Depression",test_size=0.4,batch_size=32):
    features=df.drop(columns=target_column)
    labels=df[target_column]
    
    X_train,X_val,y_train,y_val=train_test_split(features,labels,test_size=test_size,random_state=42)
    
    sc=StandardScaler()
    X_train_scaled=sc.fit_transform(X_train)
    X_val_scaled=sc.transform(X_val)
    
    train_dataset=MentalHealthDataset(X_train_scaled,y_train)
    val_dataset=MentalHealthDataset(X_val_scaled,y_val)
    
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader=DataLoader(val_dataset,batch_size=batch_size)
    
    return train_loader,val_loader,sc


# In[67]:


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_loss += loss.item()
        
        # Print epoch statistics
        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {100*train_correct/train_total:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {100*val_correct/val_total:.2f}%\n')


# In[68]:


df=df_train
train_loader,val_loader,sc=prepare_data(df)
input_size=len(df.columns)-1
model=DepressionClassifier(input_size)
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer)

