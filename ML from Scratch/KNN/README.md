# KNN FROM SCRATCH – MACHINE LEARNING FROM SCRATCH

K nearest neighbors or KNN algorithm is non-parametric, lazy learning, supervised algorithm used for classification as well as regression. KNN is often used when searching for similar items, such as finding items similar to this one. The Algorithm suggests that you are one of them because you are close to your neighbors. Now, let’s begin the article ” KNN from Scratch“.

## How does a KNN algorithm work?

To conduct grouping, the KNN algorithm uses a very basic method to perform classification. When a new example is tested, it searches at the training data and seeks the k training examples which are similar to the new test example.  It then assigns to the test example of the most similar class label.

### WHAT DOES ‘K’ IN THE KNN ALGORITHM REPRESENT?

K in KNN algorithm represents the number of nearest neighboring points that vote for a new class of test data. If k = 1, then test examples in the training set will be given the same label as the nearest example. If k = 5 is checked for the labels of the five closest classes and the label is assigned according to the majority of the voting.  

![knn-from-scratch](https://user-images.githubusercontent.com/40186859/185522861-b9325c88-31ef-459e-9b56-469d3f4df0e9.png)

### ALGORITHM

- Initialize the best value of K
- Calculate the distance between test data and trained data using Euclidean distance or any other method
- Check class categories of nearest neighbors and determine the type in which test input falls.
-  Classify the test data according to majority vote of nearest K dataset

### KNN FROM SCRATCH: MANUAL GUIDE

Let’s consider an example of height and weight 

The given dataset is about the height and weight of the customer with the respective t-shirt size where M represents the medium size and L represents the large size. Now we need to predict the t-shirt size for the new customer with a height of 169 cm and weight as 69 kg.

| Height | Weight | T-Shirt Size |
|:------:| :------:| :----------:|
| 150	| 51	| M |
| 158 |	51	| M |
| 158	 |53	| M |
| 158 |	55	| M |
| 159	| 55	| M |
| 159 |	56	| M |
| 160	| 57	| M |
| 160 |	58	| M |
| 160	| 58	| M |
| 162 |	52	| L |
| 163	| 53	| L |
| 165	| 53	| L |
| 167 | 55	| L |
| 168 |	62  |	L |
| 168 |	65	| L |
| 169	| 67	| L |
| 169 |	68	| L |
| 170 |	68	| L |
| 170	| 69	| L |

### CALCULATION 

Note: Predict the t-shirt size of new customer whose name is Sunil with height as 169cm and weight as 69 Kg. 

**Step 1**: The initial step is to calculate the Euclidean distance between the existing points and new points. For example, the existing point is (4,5) and the new point is (1, 1).

So, P1 = (4,5) where $x_1$ = 4 and $y_1$ = 5 <br>
P2 = (1,1) where $x_2$ = 1 and $y_2$ = 1

![Calcuate Euclidean Distance](https://user-images.githubusercontent.com/40186859/185524881-4f2cca3e-7952-45e6-9d72-c999b52cac27.png)

Now Euclidean distance = $\sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$ = $\sqrt{(1 - 4)^2 + (1 - 5)^2}$ = 5

**Step 2**: Now, we need to choose the k value and select the closest k neighbors to the new item. So, in our case, **K = 5**, bold elements have the least Euclidean distance as compared with others.

| Height |	Weight |	T-Shirt Size |	Distance |
| :-----: | :------: | :----------: | :--------|
| 150	| 51	| M	| 26.17 |
| 158	| 51	| M	| 21.09 |
| 158	| 53	| M	| 19.41 |
| 158	| 55	| M	| 17.80 |
| 159	| 55	| M	| 17.20 |
| 159	| 56	| M	| 16.40 |
| 160	| 57	| M	| 15    |
| 160	| 58	| M	| 14.21 |
| 160	| 58	| M	| 14.21 |
| 162	| 52	| L	| 18.38 |
| 163	| 53	| L	| 16.49 |
| 165	| 53	| L	| 16.49 |
| 167	| 55	| L	| 14.14 |
| **168**	| **62**	| **L**	| **7.01**  |
| **168**	| **65**	| **L**	| **4.12**  |
| **169**	| **67**	| **L**	| **2**     |
| **169**	| **68**	| **L**	| **2.23** |
| **170**	| **68**	| **L**	| **1.41** |
| 170	| 69	| L	| 10.04 |

**Step 3**: Since, K = 5, we have 5 t-shirts of size L. So a new customer with a height of 169 cm and a weight of 69 kg will fit into t-shirts of L size.

### BEST K-VALUE IN KNN FROM SCRATCH

K in the KNN algorithm refers to the number of nearest data to be taken for the majority of votes in predicting the class of test data. Let’s take an example how value of K matters in KNN.

![Best Value of K](https://user-images.githubusercontent.com/40186859/185526167-5f4569b3-7e51-406d-a619-807d0bcac910.png)

In the above figure, we can see that if we proceed with K=3, then we predict that test input belongs to class B, and if we continue with K=7, then we predict that test input belongs to class A.

Data scientists usually choose K as an odd number if the number of classes is 2 and another simple approach to select k is set k=sqrt(n). Similarly, you can choose the minimum value of K, find its prediction accuracy, and keep on increasing the value of K. K value with highest accuracy can be used as K value for rest of the prediction process.

![K vs Accuracy Plot](https://user-images.githubusercontent.com/40186859/185526225-d28fb3ea-412d-42ee-a0be-77ca166d6885.png)

### KNN USING SCIKIT-LEARN

The example below demonstrates KNN implementation on the iris dataset using the scikit-learn library where the iris dataset has petal length, width and sepal length, width with species class/label. Our task is to build a KNN model based on sepal and petal measurements which classify the new species.

#### STEP 1: IMPORT THE DOWNLOADED DATA AND CHECK ITS FEATURES.

```
>>> import pandas as pd
>>> iris = pd.read_csv('../dataset/iris.data', header = None)

##  attribute to return the column labels of the given Dataframe
>>> iris.columns = ["sepal_length", "sepal_width", 
...                     "petal_length", "petal_width", "target_class"]
>>> iris.dropna(how ='all', inplace = True)
>>> iris.head()
```

| sepal_length | sepal_width | petal_length | petal_width | target_class|
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| 5.1 | 3.5 | 1.4 | 0.2 | Iris-setosa |
| 4.9 | 3.0 | 1.4 | 0.2| Iris-setosa |
| 4.7 | 3.2 | 1.3 | 0.2 | Iris-setosa |
| 4.6 | 3.1 | 1.5 | 0.2 | Iris-setosa |
| 5.0 | 3.6 | 1.4 | 0.2 | Iris-setosa |

The **Iris Dataset** contains four features (length and width of sepals and petals) of 50 samples of three **species** of Iris (Iris **setosa**, Iris **virginica**, and Iris **versicolor**). Here, target_class is in a categorical form that can not be handled by machine learning algorithms like KNN. So feature and response should be numeric i.e. NumPy arrays which have a specific shape. For this, we have implemented a LabelEncoder for target_class which are encoded as 0 for iris_setosa, 1 for iris_versicolor, and 2 for iris_verginica.

#### STEP 2: SPLIT THE DATA INTO TRAIN SET AND TEST SET AND TRAIN THE KNN MODEL

It is not an optimal approach for training and testing on the same data, so we need to divide the data into two parts, the training set and testing test. For this function called ‘train_test_split’ provided by Sklearn helps to split the data where the parameter like ‘test_size’ split the percentage of train data and test data. 

‘Random_state’ is another parameter which helps to give the same result every time when we run our model means to split the data, in the same way, every time. As we are training and testing on various datasets, the subsequent quality of the tests should be a stronger approximation of how well the model would do on unknown data.  

```
## splitting the data into training and test sets 
>>> from sklearn.model_selection import train_test_split 
>>> X_train, X_test, y_train, y_test = train_test_split(data, target, 
...                                             test_size = 0.3, random_state = 524)
```

#### STEP 3: IMPORT ‘KNEIGHBORSCLASSIFIER’ CLASS FROM SKLEARN

It is important to select the appropriate value of k, so we use a loop to fit and test the model for various values for K (for 1 –  25) and measure the test accuracy of the KNN. Detail about choosing K is provided in the above KNN from scratch section.

```
## Import ‘KNeighborsClassifier’ class from sklearn
>>> from sklearn.neighbors import KNeighborsClassifier
## import metrics model to check the accuracy
>>> from sklearn import metrics
## using loop from k = 1 to k = 25 and record testing accuracy
>>> k_range = range(1,26)
>>> scores = {}
>>> score_list = []
>>> for k in k_range:
...     knn = KNeighborsClassifier(n_neighbors=k)
...     knn.fit(X_train, y_train)
...     y_pred = knn.predict(X_test)
...     scores[k] = metrics.accuracy_score(y_test, y_pred)
...     score_list.append(metrics.accuracy_score(y_test, y_pred))
```

![K Value vs Accuracy](https://user-images.githubusercontent.com/40186859/185527117-d1db5a05-de49-4830-a23b-359868fc8e89.png)

#### STEP 4: MAKE PREDICTIONS

Now we are going to choose an appropriate value of K as 5 for our model. So this is going to be our final model which is going to make predictions.

```
>>> knn = KNeighborsClassifier(n_neighbors=5)
>>> knn.fit(data,target)
>>> target_Classes = {0:'iris_setosa', 1:'iris_versicolor', 2:'iris_verginica'}
>>> x_new = [[4,3,1,2],
...         [5,4,1,3]]
>>> y_predict = knn.predict(x_new)
>>> print('First Predictions  -> ',target_Classes[y_predict[1]])
>>> print('Second Predictions -> ',target_Classes[y_predict[0]])


OUTPUT:

First Predictions  ->  iris_setosa
Second Predictions ->  iris_setosa
```

#### PROS
- KNN classifier algorithm is used to solve both regression, classification, and multi-classification problem
- KNN classifier algorithms can adapt easily to changes in real-time inputs.
- We do not have to follow any special requirements before applying KNN.

#### Cons

- KNN performs well in a limited number of input variables. So, it’s really challenging to estimate the performance of new data as the number of variables increases. Thus, it is called the curse of dimensionality. In the modern scientific era, increasing quantities of data are being produced and collected. How, for target_class in machine learning, too much data can be a bad thing. At a certain level, additional features or dimensions will decrease the precision of a model, because more data has to be generalized. Thus, this is recognized as the “Curse of dimensionality”.
- KNN requires data that is normalized and also the KNN algorithm cannot deal with the missing value problem.
- The biggest problem with the KNN from scratch is finding the correct neighbor number. 

# 💥 ESSENCE OF THE KNN ALGORITHM IN ONE PICTURE!

![132229901-06f86d02-98c2-473a-a6ce-758701bb2bc5](https://user-images.githubusercontent.com/40186859/185749018-64da0bdc-4f22-492a-a2a1-824c48309fbb.jpg)
