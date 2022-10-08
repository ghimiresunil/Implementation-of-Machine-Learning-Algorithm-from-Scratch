# 1.  K-Means Clustering From Scratch Python 

![K-Means Clustering From Scratch Python](https://user-images.githubusercontent.com/40186859/194712488-02c26271-c1ea-4c23-a667-c56d8d141827.png)


In this article, I will cover k-means clustering from scratch. In general, Clustering is defined as the grouping of data points such that the data points in a group will be similar or related to one another and different from the data points in another group. The goal of clustering is to determine the intrinsic grouping in a set of unlabelled data.

K- means is an unsupervised partitional clustering algorithm that is based on grouping data into k – numbers of clusters by determining centroid using the Euclidean or Manhattan method for distance calculation. It groups the object based on minimum distance.

![euclidean-distance-formula](https://user-images.githubusercontent.com/40186859/194710286-ba584cbb-b23c-4dfb-b046-e2c41d1204dd.png)


## 1.1. ALGORITHM

- First,  initialize the number of clusters, K (Elbow method is generally used in selecting the number of clusters )
- Randomly select the k data points for centroid. A centroid is the imaginary or real location representing the center of the cluster.
- Categorize each data items to its closest centroid and update the centroid coordinates calculating the average of items coordinates categorized in that group so far
- Repeat the process for a number of iterations till successive iterations clusters data items into the same group

## 1.2. HOW IT WORKS ?

In the beginning, the algorithm chooses k centroids in the dataset randomly after shuffling the data. Then it calculates the distance of each point to each centroid using the euclidean distance calculation method. Each centroid assigned represents a cluster and the points are assigned to the closest cluster. At the end of the first iteration, the centroid values are recalculated, usually taking the arithmetic mean of all points in the cluster.  In every iteration, new centroid values are calculated until successive iterations provide the same centroid value. 

Let’s kick off K-Means Clustering Scratch with a simple example: Suppose we have data points (1,1), (1.5,2), (3,4), (5,7), (3.5,5), (4.5,5), (3.5,4.5). Let us suppose k = 2 i.e. dataset should be grouped in two clusters. Here we are using the Euclidean distance method. 

![Working Mechanishm](https://user-images.githubusercontent.com/40186859/194710327-5028049b-338e-4195-8d3d-4910b2f05fc1.png)

**Step 01**: It is already defined that k = 2 for this problem
**Step 02**: Since k = 2, we are randomly selecting two centroid as c1(1,1) and c2(5,7)
**Step 03**: Now, we calculate the distance of each point to each centroid using the euclidean distance calculation method:

![image](https://user-images.githubusercontent.com/40186859/194710372-edd6db1c-d6b5-458a-995f-ded9e2690a36.png)

<u>**1.2.1. ITERATION 01**</u>

|X1	|Y1	|**X2**|	**Y2**	|**D1**	|X1|	Y1|	**X2**|	**Y2**|	**D2**|	Remarks|
|:------:| :------:| :----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
|1	|1	|1	|1	|0	|1	|1	|5	|7	|7.21	|D1<D2 : (1,1) belongs to c1|
|1.5	|2	|1	|1	|1.12	|1.5	|2	|5	|7	|6.1	|D1<D2 : (1.5,2) belongs to c1|
|3	|4	|1	|1	|3.61	|3	|4	|5	|7	|3.61	|D1<D2 : (3,4) belongs to c1|
|5	|7	|1	|1	|7.21	|5	|7	|5	|7	|0	|D1>D2 : (5,7) belongs to c2|
|3.5	|5	|1	|1	|4.72	|3.5	|5	|5	|7	|2.5	|D1>D2 : (3.5,5)  belongs to c2|
|4.5	|5	|1	|1	|5.32	|4.5	|5	|5	|7	|2.06	|D1>D2 : (5.5,5)  belongs to c2|
|3.5	|4.5	|1	|1	|4.3	|3.5	|4.5	|5	|7	|2.91	|D1>D2 : (3.5,4.5)  belongs to c2|

_**Note**_:  D1 & D2 are euclidean distance between centroid **(x2,y2)** and data points **(x1,y1)**

In cluster c1 we have (1,1), (1.5,2) and (3,4) whereas centroid c2 contains (5,7), (3.5,5), (4.5,5) &  (3.5,4.5). Here, a new centroid is the algebraic mean of all the data items in a cluster. 

**C1(new)** = ( (1+1.5+3)/3 , (1+2+4)/3) = **(1.83, 2.33)** <br>
**C2(new)** = ((5+3.5+4.5+3.5)/4, (7+5+5+4.5)/4) = **(4.125, 5.375)**


![Iteration 01](https://user-images.githubusercontent.com/40186859/194711026-a51bef10-dae2-4afd-9e6f-526110f31d72.png)


<u>**1.2.2. ITERATION 02**</u>

|X1	|Y1	|**X2**|	**Y2**	|**D1**	|X1|	Y1|	**X2**|	**Y2**|	**D2**|	Remarks|
|:------:| :------:| :----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
|1	|1	|1.83	|2.33	|1.56	|1	|1	|4.12	|5.37	|5.37	|(1,1) belongs to c1|
|1.5	|2	|1.83	|2.33	|0.46	|1.5	|2	|4.12	|5.37	|4.27	|(1.5,2) belongs to c1|
|3	|4	|1.83	|2.33	|2.03	|3	|4	|4.12	|5.37	|1.77	|(3,4) belongs to c2|
|5	|7	|1.83	|2.33	|5.64	|5	|7	|4.12	|5.37	|1.84	|(5,7) belongs to c2|
|3.5	|5	|1.83	|2.33	|3.14	|3.5	|5	|4.12	|5.37	|0.72	|(3.5,5)  belongs to c2|
|4.5	|5	|1.83	|2.33	|3.77	|4.5	|5	|4.12	|5.37	|0.53	|(5.5,5)  belongs to c2|
|3.5	|4.5	|1.83	|2.33	|2.73	|3.5	|4.5	|4.12	|5.37	|1.07	|(3.5,4.5)  belongs to c2|

In cluster c1 we have (1,1), (1.5,2) ) whereas centroid c2 contains (3,4),(5,7), (3.5,5), (4.5,5) &  (3.5,4.5). Here, new centroid is the algebraic mean of all the data items in a cluster. 

**C1(new)** = ( (1+1.5)/2 , (1+2)/2) = **(1.25,1.5)** <br>
**C2(new)** = ((3+5+3.5+4.5+3.5)/5, (4+7+5+5+4.5)/5) = **(3.9, 5.1)**

![Iteration 02](https://user-images.githubusercontent.com/40186859/194711267-f7864c97-442d-4267-bfdd-4b1316d9fb55.png)


<u>**1.2.3. ITERATION 03**</u>

|X1	|Y1	|**X2**|	**Y2**	|**D1**	|X1|	Y1|	**X2**|	**Y2**|	**D2**|	Remarks|
|:------:| :------:| :----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
|1	|1	|1.25	|1.5	|0.56	|1	|1	|3.9	|5.1	|5.02	|(1,1) belongs to c1|
|1.5	|2	|1.25	|1.5	|0.56	|1.5	|2	|3.9	|5.1	|3.92	|(1.5,2) belongs to c1|
|3	|4	|1.25	|1.5	|3.05	|3	|4	|3.9	|5.1	|1.42	|(3,4) belongs to c2|
|5	|7	|1.25	|1.5	|6.66	|5	|7	|3.9	|5.1	|2.19	|(5,7) belongs to c2|
|3.5	|5	|1.25	|1.5	|4.16	|3.5	|5	|3.9	|5.1	|0.41	|(3.5,5)  belongs to c2|
|4.5	|5	|1.25	|1.5	|4.77	|4.5	|5	|3.9	|5.1	|0.60	|(5.5,5)  belongs to c2|
|3.5	|4.5	|1.25	|1.5	|3.75	|3.5	|4.5	|3.9	|5.1	|0.72	|(3.5,4.5)  belongs to c2|

In cluster c1 we have (1,1), (1.5,2) ) whereas centroid c2 contains (3,4),(5,7), (3.5,5), (4.5,5) &  (3.5,4.5). Here, new centroid is the algebraic mean of all the data items in a cluster. 

**C1(new)** = ( (1+1.5)/2 , (1+2)/2) = **(1.25,1.5)** <br>
**C2(new)** = ((3+5+3.5+4.5+3.5)/5, (4+7+5+5+4.5)/5) = **(3.9, 5.1)**

**Step 04**: In the 2nd and 3rd iteration, we obtained the same centroid points. Hence clusters of above data point is :


# 2. K-Means Clustering Scratch Code

So far, we have learnt about the introduction to the K-Means algorithm. We have learnt in detail about the mathematics behind the K-means clustering algorithm and have learnt how Euclidean distance method is used in grouping the data items in K number of clusters. Here were are implementing K-means clustering from scratch using python. But the problem is how to choose the number of clusters? In this example, we are assigning the number of clusters ourselves and later we will be discussing various ways of finding the best number of clusters. 

```
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import math

class K_Means:
    
    def __init__(self, k=2, tolerance = 0.001, max_iter = 500):
        self.k = k
        self.max_iterations = max_iter
        self.tolerance = tolerance
```

We have defined a K-means class with init consisting default value of k as 2, error tolerance as 0.001, and maximum iteration as 500.
Before diving into the code, let’s remember some mathematical terms involved in K-means clustering:- centroids & euclidean distance. On a quick note centroid of a data is the average or mean of the data and Euclidean distance is the distance between two points in the coordinate plane calculated using Pythagoras theorem.

```
def euclidean_distance(self, point1, point2):
            #return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 + (point1[2]-point2[2])**2)   #sqrt((x1-x2)^2 + (y1-y2)^2)
            return np.linalg.norm(point1-point2, axis=0)
```

We find the euclidean distance from each point to all the centroids. If you look for efficiency it is better to use the NumPy function (np.linalg.norm(point1-point2, axis=0))

```
def fit(self, data):
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = data[i]
```

### ASSIGNING CENTROIDS 

There are various methods of assigning k centroid initially. Mostly used is a random selection but let’s go in the most basic way. We assign the first k points from the dataset as the initial centroids.

```
for i in range(self.max_iterations):
            
            self.classes = {}
            for j in range(self.k):
                self.classes[j] = []
                
            
            for point in data:
                distances = []
                for index in self.centroids:
                    distances.append(self.euclidean_distance(point,self.centroids[index]))
                cluster_index = distances.index(min(distances))
                self.classes[cluster_index].append(point)
```

Till now, we have defined the K-means class and initialized some default parameters. We have defined the euclidean distance calculation function and we have also assigned initial k clusters. Now, In order to know which cluster and data item belong to, we are calculating Euclidean distance from the data items to each centroid. Data item closest to the cluster belongs to that respective cluster.

```
previous = dict(self.centroids)
for cluster_index in self.classes:
    self.centroids[cluster_index] = np.average(self.classes[cluster_index], axis = 0)

isOptimal = True

for centroid in self.centroids:
    original_centroid = previous[centroid]
    curr = self.centroids[centroid]
    if np.sum((curr - original_centroid)/original_centroid * 100.0) > self.tolerance:
        isOptimal = False
if isOptimal:
    break
```

At the end of the first iteration, the centroid values are recalculated, usually taking the arithmetic mean of all points in the cluster. In every iteration, new centroid values are calculated until successive iterations provide the same centroid value.

<u>**CLUSTERING WITH DEMO DATA**</u>

We’ve now completed the K Means scratch code of this Machine Learning tutorial series. Now, let’s test our code by clustering with randomly generated data:

```
#generate dummy cluster datasets
# Set three centers, the model should predict similar results
center_1 = np.array([1,1])
center_2 = np.array([5,5])
center_3 = np.array([8,1])

# Generate random data and center it to the three centers
cluster_1 = np.random.randn(100, 2) + center_1
cluster_2 = np.random.randn(100,2) + center_2
cluster_3 = np.random.randn(100,2) + center_3

data = np.concatenate((cluster_1, cluster_2, cluster_3), axis = 0)
```

Here we have created 3 groups of data of two-dimension with a different centre. We have defined the value of k as 3. Now, let’s fit the model created from scratch.

```
k_means = K_Means(K)
k_means.fit(data)


# Plotting starts here
colors = 10*["r", "g", "c", "b", "k"]

for centroid in k_means.centroids:
    plt.scatter(k_means.centroids[centroid][0], k_means.centroids[centroid][1], s = 130, marker = "x")

for cluster_index in k_means.classes:
    color = colors[cluster_index]
    for features in k_means.classes[cluster_index]:
        plt.scatter(features[0], features[1], color = color,s = 30
```
![K Means Centroid Graph](https://user-images.githubusercontent.com/40186859/194712252-d5091577-cce6-4d1f-b398-c3f1351f410a.png)

## 3. K-MEANS USING SCIKIT-LEARN

```
from sklearn.cluster import KMeans
center_1 = np.array([1,1])
center_2 = np.array([5,5])
center_3 = np.array([8,1])

# Generate random data and center it to the three centers
cluster_1 = np.random.randn(100,2) + center_1
cluster_2 = np.random.randn(100,2) + center_2
cluster_3 = np.random.randn(100,2) + center_3

data = np.concatenate((cluster_1, cluster_2, cluster_3), axis = 0)
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)
plt.scatter(data[:,0],data[:,1], c=kmeans.labels_, cmap='rainbow')
```

![image](https://user-images.githubusercontent.com/40186859/194712283-56421dbf-a972-4fac-b0a5-bf74d7c6bdcd.png)


### CHOOSING VALUE OF K 

While working with the k-means clustering scratch, one thing we must keep in mind is the number of clusters ‘k’. We should make sure that we are choosing the optimum number of clusters for the given data set.  But, here arises a question, how to choose the optimum value of k ?? We use the elbow method which is generally used in analyzing the optimum value of k.

The Elbow method is based on the principle that **“Sum of squares of distances of every data point from its corresponding cluster centroid should be as minimum as possible”. **

### STEPS OF CHOOSING BEST K VALUE

- Run k-means clustering model on various values of k 
- For each value of K, calculate the Sum of squares of distances of every data point from its corresponding cluster centroid which is called WCSS ( Within-Cluster Sums of Squares)
- Plot the value of WCSS with respect to various values of K
- To select the value of k, we choose the value where there is bend (knee) on the plot i.e. WCSS isn’t increasing rapidly.

![Elbow Method to find Value of K](https://user-images.githubusercontent.com/40186859/194712397-09dae5d8-1b45-48e1-a3e1-4a6bb528b3c3.png)

### PROS OF K-MEANS 
- Relatively simple to learn and understand as the algorithm solely depends on the euclidean method of distance calculation. 
- K means works on minimizing Sum of squares of distances, hence it guarantees convergence
- Computational cost is O(K*n*d), hence K means is fast and efficient


### CONS OF K-MEANS
- Difficulty in choosing the optimum number of clusters K
- K means has a problem when clusters are of different size, densities, and non-globular shapes
- K means has problems when data contains outliers
- As the number of dimensions increases, the difficulty in getting the algorithm to converge increases due to the curse of dimensionality
- If there is overlapping between clusters, k-means doesn’t have an intrinsic measure for uncertainty
