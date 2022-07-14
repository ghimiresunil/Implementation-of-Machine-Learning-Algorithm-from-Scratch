# 1. NAIVE BAYES ALGORITHM FROM SCRATCH

Naive Bayes is a classification algorithm based on the “Bayes Theorem”. So let’s get introduced to the Bayes Theorem first.

![Naive Bayes Intro](https://user-images.githubusercontent.com/40186859/178624935-599a6618-4c3a-47d6-9232-4df5a673712c.jpg)

Bayes Theorem is used to find the probability of an event occurring given the probability of another event that has already occurred. Here _**B**_ is the evidence and _**A**_ is the hypothesis. Here _**P(A)**_ is known as prior, _**P(A/B)**_ is posterior, and _**P(B/A)**_ is the likelihood.

_**Posterior**_ = $\frac{prior \times likelihood} {evidence}$

## 1.1 NAIVE BAYES ALGORITHM

The name Naive is used because the presence of one independent feature doesn’t affect (influence or change the value of) other features. The most important assumption that Naive Bayes makes is that all the features are independent of each other. Being less prone to overfitting, Naive Bayes algorithm works on Bayes theorem to predict unknown data sets.

**EXAMPLE**:

| AGE | INCOME | STUDENT | CREDIT | BUY COMPUTER |
| :-: | :----: | :-----: | :----: | :----------: |
| Youth |	High |	No	| Fair |	No |
| Youth | High |	No	| Excellent |	No |
| Middle Age |	High |	No |	Fair | Yes |
| Senior |	Medium |	No |	Fair |	Yes |
| Senior |	Low	| Yes	| Fair	| Yes |
| Senior |	Low |	Yes |	Excellent |	No |
| Middle Age	| Low |	Yes	| Excellent |	Yes |
| Youth |	Medium |	No |	Fair |	No |
| Youth	| Low	| Yes	| Fair	| Yes |
| Senior |	Medium |	Yes |	Fair |	Yes |
| Youth |	Medium |	Yes |	Excellent |	Yes |
| Middle Age |	Medium |	No |	Excellent |	Yes |
| Middle Age	| High |	Yes	| Fair |	Yes |
| Senior |	Medium |	No |	Excellent |	No |

We are given a table that contains a dataset about **age**, **income**, **student**, **credit-rating**, **buying a computer**, and their respective features. From the above dataset, we need to find whether a youth student with medium income having a fair credit rating buys a computer or not. 

i.e. B = (Youth, Medium, Yes, Flair)

In the above dataset, we can apply the Bayesian theorem. 

P(A|B) = $\frac{P(B|A) \times P(A)}{P(B)}$

Where, <br>
**A** = ( Yes / No ) under buying computer <br>
**B** = ( Youth, Medium, Student, Fair)

So, **P(A/B)** means the probability of buying a computer given that conditions are “**Youth age**”, “**Medium Income**”, “**Student**”, and “**fair credit-rating**”. 

**ASSUMPTION**:

Before starting, we assume that all the given features are independent of each other.

### STEP 1: CALCULATE PROBABILITIES OF BUYING A COMPUTER FROM ABOVE DATASET

| Buy Computer | Count	| Probability |
| :----------: | :------: | :--------: |
| Yes |	9 | 9/14 |
| No | 5 |	5/14 |
|Total | 14 | |

### STEP 2: CALCULATE PROBABILITIES UNDER CREDIT-RATING BUYING A COMPUTER FROM THE ABOVE DATASET

![image](https://user-images.githubusercontent.com/40186859/178889592-4b0e84fe-6f5a-4dbf-ae27-e5742797d3cd.png)

Let’s understand how we calculated the above probabilities. From the table we can see that there are 8 fair credit ratings among which 6 buy computers and 2 don’t buy. Similarly, 6 have excellent credit ratings among which 3 buy computers and 3 don’t. As a whole 9 (6+3)  buy computers and 5 (2+5) don’t.

P(fair / Yes) means the probability of credit rating being fair when someone buys a computer. Hence, P(fair / Yes) = P( fair buying computer) / P ( total number of buying computer) i.e. 6/9.

### STEP 3: CALCULATE PROBABILITIES UNDER STUDENT BUYING A COMPUTER FROM THE ABOVE DATASET

![image](https://user-images.githubusercontent.com/40186859/178890224-3b2a6c32-a15e-4579-95b1-8df38dfd0296.png)

#### STEP 4: CALCULATE PROBABILITIES UNDER INCOME LEVEL  BUYING A COMPUTER FROM THE ABOVE DATASET

P( High / Yes ) = 2/9 <br>
P( Mid / Yes ) = 4/9 <br>
P( Low / Yes ) = 3/9 <br>
P( High / No ) = 2/5 <br>
P( Mid / No ) = 2/5 <br>
P( Low / No ) =  1/5 <br>

### STEP 5: CALCULATE PROBABILITIES UNDER AGE LEVEL  BUYING A COMPUTER FROM THE ABOVE DATASET

P( Youth / Yes ) = 2/9 <br>
P( Mid / Yes ) = 4/9 <br>
P( Senior / Yes ) = 3/9 <br>
P( Youth / No ) = 3/5 <br>
P( Mid / No ) = 0 <br>
P( Senior / No ) = 2/5 <br>

**CALCULATION**

We have,
B = ( Youth, Medium, Student, Fair)

```
P(Yes) * P(B / Yes )  = P(Yes) * P( Youth / Yes ) * P( Mid / Yes) * P( S Yes / Yes) * P( Fair / Yes) 
                      =  9/14 * 2/9 * 4/9 * 6/9 * 6/9
                      = 0.02821
P(No) * P(B / No ) = P(No) * P( Youth / No ) * P( Mid / No) * P( S Yes / No) * P( Fair / No)
                   = 5/14 * 3/5 * 2/5 * 1/5 * 2/5
                   = 0.04373
```
Hence,

```
P(Yes / B) = P(Yes) * P(B / Yes )  / P(B)
           = 0.02821 / 0.04373
           = 0.645
           
P(No / B) =  P(No) * P(B / No )  / P(B)
          = 0.0068 / 0.04373
          = 0.155
```

Here,  P(Yes / B) is greater than  P(No / B) i.e posterior Yes is greater than posterior No. So the class B ( Youth, Mid, yes, fair) buys a computer.

## 1.2. NAIVE BAYES FROM SCRATCH

** Classify whether a given person is a male or a female based on the measured features. The features include height, weight, and foot size.
**

![image](https://user-images.githubusercontent.com/40186859/178891887-dff0526c-1986-4935-ae1e-4e4c79370dcb.png)

Now, defining a dataframe which consists if above provided data.
```
import pandas as pd
import numpy as np

# Create an empty dataframe
data = pd.DataFrame()

# Create our target variable
data['Gender'] = ['male','male','male','male','female','female','female','female']

# Create our feature variables
data['Height'] = [6,5.92,5.58,5.92,5,5.5,5.42,5.75]
data['Weight'] = [180,190,170,165,100,150,130,150]
data['Foot_Size'] = [12,11,12,10,6,8,7,9]
```

Creating another data frame containing the feature value of height as 6 feet, weight as 130 lbs and foot size as 8 inches. using Naive Bayes from scratch we are trying to find whether the gender is male or female.

```
# Create an empty dataframe
person = pd.DataFrame()

# Create some feature values for this single row
person['Height'] = [6]
person['Weight'] = [130]
person['Foot_Size'] = [8]
```

Calculating the total number of males and females and their probabilities i.e priors:

```
# Number of males
n_male = data['Gender'][data['Gender'] == 'male'].count()

# Number of males
n_female = data['Gender'][data['Gender'] == 'female'].count()

# Total rows
total_ppl = data['Gender'].count()

# Number of males divided by the total rows
P_male = n_male/total_ppl

# Number of females divided by the total rows
P_female = n_female/total_ppl
```
Calculating mean and variance of male and female of the feature height, weight and foot size.

```
# Group the data by gender and calculate the means of each feature
data_means = data.groupby('Gender').mean()

# Group the data by gender and calculate the variance of each feature
data_variance = data.groupby('Gender').var()
```
![image](https://user-images.githubusercontent.com/40186859/178892000-88241b29-59f7-4215-bfb2-0ddbbc276eb7.png)

**Formula**:

* posterior (male) = P(male)*P(height|male)*P(weight|male)*P(foot size|male) / evidence
* posterior (female) = P(female)*P(height|female)*P(weight|female)*P(foot size|female) / evidence
* Evidence = P(male)*P(height|male)*P(weight|male)*P(foot size|male) + P(female) * P(height|female) * P(weight|female)*P(foot size|female)

The evidence may be ignored since it is a positive constant. (Normal distributions are always positive.)

**CALCULATION**:

![image](https://user-images.githubusercontent.com/40186859/178892160-db8c26c8-1207-4c58-a217-1e7de9ea05d5.png)

**Calculation of P(height | Male )**

mean of  the height of male = 5.855 <br>
variance ( Square of S.D.)  of the height of a male is square of 3.5033e-02 and x i.e. given height is 6 feet <br>
Substituting the values in the above equation we get  P(height | Male ) = 1.5789

```
# Create a function that calculates p(x | y):
def p_x_given_y(x, mean_y, variance_y):

    # Input the arguments into a probability density function
    p = 1/(np.sqrt(2*np.pi*variance_y)) * np.exp((-(x-mean_y)**2)/(2*variance_y))
    
    # return p
    return p
```

Similarly,

P(weight|male) = 5.9881e-06 <br>
P(foot size|male) = 1.3112e-3 <br>
P(height|female) = 2.2346e-1 <br>
P(weight|female) = 1.6789e-2 <br>
P(foot size|female) = 2.8669e-1 <br>

Posterior (male)*evidence = P(male)*P(height|male)*P(weight|male)*P(foot size|male) = 6.1984e-09 <br>
Posterior (female)*evidence = P(female)*P(height|female)*P(weight|female)*P(foot size|female)= 5.3778e-04

**CONCLUSION**
Since Posterior (female)*evidence > Posterior (male)*evidence, the sample is female.

## 1.3. NAIVE BAYES USING SCIKIT-LEARN

```
import pandas as pd
import numpy as np

# Create an empty dataframe
data = pd.DataFrame()

# Create our target variable
data['Gender'] = [1,1,1,1,0,0,0,0]   #1 is male
# Create our feature variables
data['Height'] = [6,5.92,5.58,5.92,5,5.5,5.42,5.75]
data['Weight'] = [180,190,170,165,100,150,130,150]
data['Foot_Size'] = [12,11,12,10,6,8,7,9]
```

Though we have very small dataset, we are dividing the dataset into train and test do that it can be used in other model prediction. We are importing gnb() from sklearn and we are training the model with out dataset.

```
X = data.drop(['Gender'],axis=1) 
y=data.Gender

  
# splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
  
# training the model on training set
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
  
# making predictions on the testing set
y_pred = gnb.predict(X_test)
```

Now, our model is ready. Let’s use this model to predict on new data.

```
# Create our target variable
data1 = pd.DataFrame()

# Create our feature variables
data1['Height'] = [6]
data1['Weight'] = [130]
data1['Foot_Size'] = [8]

y_pred = gnb.predict(data1)
if y_pred==0:
    print ("female")
else:
    print ("male")

Output: Female
```

**CONCLUSION**

We have come to an end of Naive Bayes from Scratch. If you have any queries, feedback, or suggestions then you can leave a comment or mail on **info@sunilghimire.com.np**. See you in the next tutorial. 

### Stay safe !! Happy Coding !!!
