For the sake of simplicity, let’s assume our multi-class classification problem to be a 3-class classification problem. Say we have got three class labels(Target/Outcome) in a dataset, namely 0, 1, and 2. A potential uncertainty matrix for these groups is provided below.

![Multi Class Confusion Matrix](https://user-images.githubusercontent.com/40186859/177247062-b7c0826c-db05-4a04-a6bf-8a2e8d56a782.png)

Unlike binary classification, there are no positive or negative classes here. At first, it might be a little difficult to find TP, TN, FP, and FN since there are no positive or negative classes, but it’s actually pretty easy

## 1. Accuracy

Accuracy is the most commonly used matrix to evaluate the model which is actually not a clear indicator of the performance.

Accuracy = $\frac{34 + 52 + 33}{150} =  0.793$

## 2. Misclassification Rate/ Error

Error tells you what fraction of predictions were incorrect. It is also known as Classification Error.

Misclassification Rate/ Error = 1 - Accuracy = 1 - 0.793 = 0.207

## 3. Precision / Positive Predicted Value

Precision is the percentage of positive instances out of the total predicted positive instances which means precision or positive predicted value means how much model is right when it says it is right.

* Precision for Class 0 = $\frac{34}{34 + 0 + 13} = 0.723$
 
* Precision for Class 1 = $\frac{52}{13 + 52 + 0} = 0.8$

* Precision for Class 2 = $\frac{33}{5 + 0 + 33} = 0.868$

## 4. Recall / True Positive Rate / Sensitivity

Recall literally is how many of the true positives were recalled (found), i.e. how many of the correct hits were also found.

* Recall for Class 0 = $\frac{34}{34 + 13 + 5} = 0.653$

* Recall for Class 1 = $\frac{52}{0 + 52 + 0} = 1$

* Recall for class 2 = $\frac{33}{13 + 0 + 33} = 0.7173$

## 5. F1-Score

F1- Score is the harmonic mean of the precision and recall which means the higher the value of the f1-score better will be the model. Due to the product in the numerator, if one goes low, the final F1 score goes down significantly. So a model does well in F1 score if the positive predicted are actually positives (precision) and doesn't miss out on positives and predicts them negative (recall).

* F1- Score of Class 0 = $\frac{2 \times R_0 \times P_0}{R_0 + P_0} = \frac{2 \times 0.723 \times 0.653}{0.7223 + 0.6563} = 0.6886$

* F1- Score of Class 1 = $\frac{2 \times R_1 \times P_1}{R_1 + P_1} = \frac{2 \times 0.8 \times 1}{0.8 + 1} = 0.8888$

* F1- Score of Class 2 = $\frac{2 \times R_2 \times P_2}{R_2 + P_2} = \frac{2 \times 0.868 \times 0.7170}{0.868 + 0.7170} = 0.785$

## 6. Support

 The support is the number of occurrences of each particular class in the true responses (responses in your test set). Support can calculate by summing the rows of the confusion matrix.
 
Here, the support for classes 0, 1, and 2 is 52, 52, and 46.

* Support for Class 0 = 52 
* Support for Class 1 = 52
* Support for Class 2 = 46

## 7. Micro F1

This is called a micro-averaged F1-score. It is calculated by considering the total TP, total FP and total FN of the model. It does not consider each class individually, It calculates the metrics globally. So for our example,

* Total TP = 34 + 52 + 33 = 119
* Total FP = (0 + 13) + (13 + 0) + (5 + 0) = 31
* Total FN = (13 + 5) + (0 + 0) + (13+0) = 31

Hence,

* $Micro_{Recall} = \frac{TP}{TP + FN} = \frac{119}{119 + 31} = 0.793 $

* $Micro_{Precision} = \frac{TP}{TP + FP} = \frac{119}{119 + 31} = 0.793 $

Now we can use the regular formula for F1-score and get the Micro F1-score using the above precision and recall.

* Micro F1-Score = $\frac{2 \times Micro_{Recall} \times Micro_{Precision}}{ Micro_{Recall} +  Micro_{Precision}} = \frac{2 \times 0.793 \times 0.793}{0.793 + 0.793} = 0.793$

_**As you can see When we are calculating the metrics globally all the measures become equal. Also if you calculate accuracy you will see that,**_

```
Precision = Recall = Micro F1 = Accuracy
```

## 8. Macro F1

This is macro-averaged F1-score. It calculates metrics for each class individually and then takes unweighted mean of the measures. As we have seen “Precision, Recall and F1-score for each Class.

| Precision | Recall | F1-Score |
| --------- | -------| ---------| 
| Class 0 Precision = 0.723 | Class 0 Recall= 0.653 | Class 0 F1-Score = 0.686 |
| Class 1 Precision = 0.8 | Class 1 Recall=1 | Class 1 F1-Score = 0.8888 |
| Class 2 Precision = 0.868 | Class 2 Recall= 0.7173 | Class 2 F1-Score = 0.785 |


Hence,

* Macro Average for Precision = $\frac{0.723 + 0.8 + 0.868}{3} = 0.797$

* Macro Average for Recall = $\frac{0.653 + 1 + 0.7173}{3} = 0.7901$

* Macro Average for F1-Score = $\frac{0.686 + 0.8888 + 0.785}{3} = 0.7866$

## 9. Weighted Average

Weighted Average is the method of calculating a kind of arithmetic mean of a set of numbers in which some elements of the set have greater (weight) value than others. Unlike Macro F1, it takes a weighted mean of the measures. The weights for each class are the total number of samples of that class.

* Weighted Average for precision = $\frac{0.723 \times 47 + 0.8 \times 65 + 0.868 \times 38}{150} = 0.7931$

* Weighted Average for Recall = $\frac{0.653 \times 52 + 1 \times 52 + 0.7173 \times 46}{150} = 0.79301$

* Weighted Average for F1-Score = $\frac{2 \times 𝑊𝑒𝑖𝑔ℎ𝑡𝑒𝑑_{𝑅𝑒𝑐𝑎𝑙𝑙} \times 𝑊𝑒𝑖𝑔ℎ𝑡𝑒𝑑_{𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛}}{150} = 0.79305 $

**Finally, let’s look generated Confusion matrix using Python’s Scikit-Learn**
![sklearn_confusion_matrix](https://user-images.githubusercontent.com/40186859/177247201-b6109ed7-3cea-4bed-946d-6cd4e183be44.png)

