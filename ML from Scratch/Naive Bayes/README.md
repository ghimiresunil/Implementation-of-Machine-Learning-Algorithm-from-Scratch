# 1. NAIVE BAYES ALGORITHM FROM SCRATCH

Naive Bayes is a classification algorithm based on the “Bayes Theorem”. So let’s get introduced to the Bayes Theorem first.

![Naive Bayes Intro](https://user-images.githubusercontent.com/40186859/178624935-599a6618-4c3a-47d6-9232-4df5a673712c.jpg)

Bayes Theorem is used to find the probability of an event occurring given the probability of another event that has already occurred. Here _**B**_ is the evidence and _**A**_ is the hypothesis. Here _**P(A)**_ is known as prior, _**P(A/B)**_ is posterior, and _**P(B/A)**_ is the likelihood.

_**Posterior**_ = $\frac{prior \times likelihood} {evidence}$

## 1.1 NAIVE BAYES ALGORITHM

The name Naive is used because the presence of one independent feature doesn’t affect (influence or change the value of) other features. The most important assumption that Naive Bayes makes is that all the features are independent of each other. Being less prone to overfitting, Naive Bayes algorithm works on Bayes theorem to predict unknown data sets.






