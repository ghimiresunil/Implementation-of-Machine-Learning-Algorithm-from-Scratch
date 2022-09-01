# Support Vector Machine – SVM From Scratch Python

![Support Vector Machine – SVM From Scratch Python](https://user-images.githubusercontent.com/40186859/186307013-a414ceb5-c413-47c8-8979-b5679160cc80.png)

In the 1960s, Support vector Machine (SVM) known as supervised machine learning classification was first developed, and later refined in the 1990s which has become extremely popular nowadays owing to its extremely efficient results. The SVM is a supervised algorithm is capable of performing classification, regression, and outlier detection. But, it is widely used in classification objectives. SVM is known as a fast and dependable classification algorithm that performs well even on less amount of data. Let’s begin today’s tutorial on SVM from scratch python.

![SVM](https://user-images.githubusercontent.com/40186859/186307570-c7c4c74f-c1ee-4b9a-bb27-c8bc84a740a7.png)

## HOW SVM WORKS ?

SVM finds the best N-dimensional hyperplane in space that classifies the data points into distinct classes. Support Vector Machines uses the concept of ‘Support Vectors‘, which are the closest points to the hyperplane. A hyperplane is constructed in such a way that distance to the nearest element(support vectors) is the largest. The better the gap, the better the classifier works.

<img src="https://user-images.githubusercontent.com/40186859/186307734-76e3ac6c-85de-45a3-9a43-be8da8d8cf7f.png" alt = "Selecting Hyperplane with Greater Gap" width="450">

The line (in 2 input feature) or plane (in 3 input feature) is known as a decision boundary. Every new data from test data will be classified according to this decision boundary. The equation of the hyperplane in the ‘M’ dimension:

$y = w_0 + w_1x_1 + w_2x_2 + w_3x_3 + ......= w_0 + \Sigma_{i=1}^mw_ix_i = w_0 + w^{T}X = b + w^{T}X $

where, <br>
$W_i$ = $vectors\(w_0, w_1, w_2, w_3, ...... w_m\)$

![Hyperplane Function h](https://user-images.githubusercontent.com/40186859/187906252-5809efe7-c7c6-4c51-877f-f4e1cb23d545.png)

The point above or on the hyperplane will be classified as class +1, and the point below the hyperplane will be classified as class -1.

![Support Vector](https://user-images.githubusercontent.com/40186859/187906410-defb5694-152e-4094-a25c-a6ad11b02c0e.png)

## SVM IN NON-LINEAR DATA

SVM can also conduct non-linear classification.

![SVM in Linear Data](https://user-images.githubusercontent.com/40186859/187906503-26558f78-7f0c-4837-9053-32fd2adc755c.png)

For the above dataset, it is obvious that it is not possible to draw a linear margin to divide the data sets. In such cases, we use the kernel concept.

SVM works on mapping data to higher dimensional feature space so that data points can be categorized even when the data aren’t otherwise linearly separable. SVM finds mapping function to convert 2D input space into 3D output space. In the above condition, we start by adding Y-axis with an idea of moving dataset into higher dimension.. So, we can draw a graph where the y-axis will be the square of data points of the X-axis.

![Increasing Dimension of Data](https://user-images.githubusercontent.com/40186859/187906570-1ca7370d-3065-4345-a58d-0c65a0c05b15.png)

And now, the data are two dimensional, we can draw a Support Vector Classifier that classifies the dataset into two distinct regions. Now, let’s draw a support vector classifier.

![upport Vector Classifier](https://user-images.githubusercontent.com/40186859/187906664-46375425-a7f2-4a4e-86cb-0a9c6fdba9cf.png)

Note: This example is taken from [Statquest](https://www.youtube.com/watch?v=efR1C6CvhmE).

## HOW TO TRANSFORM DATA ??


