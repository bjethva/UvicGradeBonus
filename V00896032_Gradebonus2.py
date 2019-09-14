#!/usr/bin/env python
# coding: utf-8

# In[140]:


#Clustering with K-means.cross-tabulation method to check accuracy of clustering.
#Different Visualization techniques like dendogram and t-SNE and brief of dimension reduction techniques.
#clustering finds patterns in data and put new samples to the new cluster groups accordingly
#Here, i have implemented K-means on grain dataset which remembers the mean of each cluster. And new samples are assigned whose centroid is closest.

#Importing the necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# In[141]:


#Reading csv file and printing first five samples of the data
data_full = pd.read_csv("D:/578D/GradeBonus/seeds.csv",header=None)
data = data_full.drop(data_full.columns[7],axis=1)
data.head()
#we have 3 different grain varieties: Kama, Rosa and Canadian. which is not mentioned in dataset given
#Below are the labels of the dataset which only will be used for comparison not for prediction
varieties = data_full.iloc[:,7]
varieties = np.array(varieties)


# In[142]:


#Here we dont know how many clusters will be good for this dataset we can chooes good number of clustering using K-means Inertia
#Graph. Inertia is a distance of point from the center of the cluster

#here we can see from the below plot that inercia has started decreasing slowly from number of cluster point 3 and as per the 
#analysis point is good where it starts decreasing slowly so 3 clusters should be good for this dataset

#Taking random values from 1 to 10
ks = range(1, 10)
#creating array of inertias 
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(data)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot number of clusters vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
print(inertias)


# In[143]:


#fitting the data to the model
model = KMeans(n_clusters=3)

# Use fit_predict to fit model and obtain cluster labels
labels = model.fit_predict(data)

# Create a DataFrame with labels and varieties as columns so that we can recombine the data and check which labels has been 
#correctly predicted
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab to count the number of times each grain variety coincides with each cluster label
ct = pd.crosstab(df['labels'],df['varieties'])
print(ct)


# In[144]:


#using linkage function of scipy to do hierarchical clustering of grain samples and using dendogram to visualize the result
#dendogram groups the clusters in larger groups as we go up in y-axis
#At begining each grain sample is an individual cluster and each step two closest clusters will be merged. it will continue 
#until one single cluster is left #here y axis in dendogram is distance between merging cluster
#calculating the linkage
#here i'm using complex method which means distance between clusters is max(distance between their samples)
l1 = linkage(data,method='complete')

# Plotting the dendrogram
dendrogram(l1,labels=varieties)
plt.show()


# In[145]:


#here i'm using another data visualization technique for seeds dataset which is not covered in class
#t-SNE (t-distributed stochastic neighbour embedding)
#t-SNE maps samples from high dimentional space to 2 or 3 dimensional space
#setting the model with learning rate of 200
model = TSNE(learning_rate=100)
op1 = model.fit_transform(data)
# Select the output features
x1 = op1[:,0]
y1 = op1[:,1]

# Scatter plot, coloring by varieties
plt.scatter(xs, ys, c=varieties)
plt.show()
#here we can see that all seven features can be plotted in 2-D as 


# In[149]:


#dimension reduction techniques
#In real world we need to deal with more numbers of features and more numbers features are headache for clustering. 
#so there are some dimension reduction techniques for noisy features one of them is PCA
#PCA stands for principal component analysis which is fundamental dimension reduction technique.
#it finds significant features with high variance
#pca sets all features in descending order of significant variance.
#first step it rotates the sample so that it will be alligned with axes and it also shifts the samples
#so that they will have mean of zero
pca = PCA()
pca.fit(data)
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()


# In[150]:


#here, if we do rescalling of the data and post that it is provided to model then we can get better results. as we can see from the 
#graph that second feature variance has increased a little bit
#from the graph we can say that dataset has 3 intrinsic features with high variance. so dataset features can be reduced to three.

pipeline = make_pipeline(scaler, pca)
pipeline.fit(data)
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()

