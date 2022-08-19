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

