#Clustering and reduction of dimensions

#clustering finds patterns in data and put new samples to the new cluster groups accordingly
#Here, i have implemented K-means which remembers the mean of each cluster. And new samples are assigned whose centroid is closest.

#Importing the necessary Libraries
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

#from sklearn.pipeline import make_pipeline
#from sklearn.preprocessing import StandardScaler

#Reading csv file and printing first five samples of the data
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("D:/578D/GradeBonus/fish.csv",header=None)
data.head()

#First question is howmany clusters are required. we have 4 different type of fish here so there will be 4 clusters
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=4)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler,kmeans)

#create a cross-tabulation to compare the cluster labels with the fish species.

pipeline.fit(samples)

# Calculate the cluster labels: labels
labels = pipeline.predict(samples)

# Create a DataFrame with labels and species as columns: df
df = pd.DataFrame({'labels': labels,'species': species})

# Create crosstab: ct
ct = pd.crosstab(df['labels'],df['species'])

# Display ct
print(ct)
