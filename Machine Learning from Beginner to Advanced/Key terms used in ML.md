# KEY TERMS USED IN MACHINE LEARNING

Before starting tutorials on machine learning, we came with an idea of providing a brief definition of key terms used in Machine Learning. These key terms will be regularly used in our coming tutorials on machine learning from scratch and will also be used in further higher courses. So let’s start with the term Machine Learning:

## MACHINE LEARNING
Machine learning is the study of computer algorithms that comprises of algorithms and statistical models that allow computer programs to automatically improve through experience. It is the science of getting computers to act by feeding them data and letting them learn a few tricks on their own without being explicitly programmed.

> TYPES OF MACHINE LEARNING <br>
> **Do you want to know about supervised, unsupervised and reinforcement learning?** [Read More](https://github.com/ghimiresunil/Implementation-of-Machine-Learning-Algorithm-from-Scratch/blob/main/Machine%20Learning%20from%20Beginner%20to%20Advanced/Introduction%20to%20ML%20and%20AI.md)

## CLASSIFICATION
Classification, a sub-category of supervised learning, is defined as the process of separating data into distinct categories or classes. These models are built by providing a labeled dataset and making the algorithm learn so that it can predict the class when new data is provided. The most popular classification algorithms are Decision Tree, SVM. 

We have two types of learners in respective classification problems:

* Lazy Learners: As the name suggests, such kind of learners waits for the testing data to be appeared after storing the training data. Classification is done only after getting the testing data. They spend less time on training but more time on predicting. Examples of lazy learners are K-nearest neighbor and case-based reasoning.

* Eager Learners: As opposite to lazy learners, eager learners construct a classification model without waiting for the testing data to be appeared after storing the training data. They spend more time on training but less time on predicting. Examples of eager learners are Decision Trees, Naïve Bayes, and Artificial Neural Networks (ANN).

**_Note_**: We will study these algorithms in the coming tutorials.

## REGRESSION
While classification deals with predicting discrete classes, regression is used in predicting continuous numerical valued classes. Regression is also falls under supervised learning generally used to answer “How much?” or “How many?”. Regressions create relationships and correlations between different types of data. Linear Regression is the most common regression algorithm.

Regression models are of two types:

* Simple regression model: This is the most basic regression model in which predictions are formed from a single, univariate feature of the data.
* Multiple regression model: As the name implies, in this regression model the predictions are formed from multiple features of the data. 
## CLUSTERING
Cluster is defined as groups of data points such that data points in a group will be similar or related to one another and different from the data points of another group. And the process is known as clustering. The goal of clustering is to determine the intrinsic grouping in a set of unlabelled data. Clustering is a form of unsupervised learning since it doesn’t require labeled data.

## DIMENSIONALITY
The dimensionality of a data set is the number of attributes or features that the objects in the dataset have. In a particular dataset, if there are number of attributes, then it can be difficult to analyze such dataset which is known as curse of dimensionality.

## CURSE OF DIMENSIONALITY
Data analysis becomes difficult as the dimensionality of data set increases. As dimensionality increases, the data becomes increasingly sparse in the space that it occupies.

* For classification, there will not be enough data objects to allow the creation of model that reliably assigns a class to all possible objects.
* For clustering, the density and distance between points which are critical for clustering becomes less meaningful.
![curse_of_dimensionality](https://user-images.githubusercontent.com/40186859/177011136-456ef7bd-5e74-4da2-9ed0-ed476ed1755b.jpg)


## UNDERFITTING
A machine learning algorithm is said to have underfitting when it can’t capture the underlying trend of data. It means that our model doesn’t fit the data well enough. It usually happens when we have fewer data to build a model and also when we try to build a linear model with non-linear data when using a less complex model.

## OVERFITTING
When a model gets trained with so much data, it starts learning from the noise and inaccurate data entries in our dataset. The cause of overfitting is non-parametric and non-linear methods. We use cross-validation to reduce overfitting which allows you to tune hyperparameters with only your original training set. This allows you to keep your test set as truly unseen dataset for selecting your final model.

![data](https://user-images.githubusercontent.com/40186859/177011180-3fd7849f-0dc2-419a-8bca-1ff115b6e535.png)

