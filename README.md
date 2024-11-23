# CRICKET ANALYTICS: PREDICTING SHOT EFFECTIVENESS

## ABSTRACT 

Shot efficiency in cricket batting depends on complex movement patterns of important body parts. When 
a significant amount of ball-by-ball data is available, supervised machine learning models can predict shot 
effectiveness. An automated approach to batting evaluation in cricket could be useful in applications such 
as performance evaluation, talent identification and injury prevention. Current motion evaluation 
and stroke execution are typically performed in an artificial environment with camera-based motion 
tracking systems to collect motion data, which requires careful preparation, data collection, and post
processing, and compromises the hitter's natural game. By training a model based on the exact results of 
cricket innings, supervised machine learning may be able to reliably predict cricket batting shots 
effectiveness. Three different models were tested on the training set: Decision Tree Regressor, Random 
Forest Regressor, and XGBoost Regressor. The best performing model from this group was the XGBoost 
Regressor. An exhaustive grid search was done to tune the hyperparameter and refine the model. The final 
model reported an R-squared value of 0.918, which means your model is capable to explain 91.8 per cent 
of the variance of data. This shows that the effectiveness of a cricket shot can be predicted better-than
chance but still has many unpredictable factors.

## Introduction 

Being able to predict the effectiveness of the shot a batsman will attempt, given the situation of the ball and 
the match, is one of the most difficult and strategically important tasks in cricket. The aim of every batsman 
is to score as many runs as possible without being dismissed. Although simple in principle, the format of 
the game greatly affects the stroke type and style of the batsman. In shorter game formats like T20 and One 
Day Internationals, batsmen tend to be more aggressive as their team has a limited number of balls to score 
from (120 and 300 balls respectively).  
Getting the right batsman vs bowler match-up is of paramount importance. For example, for a fielding team, 
the choice of bowler against the opposition's star batsman can be the key difference between winning or 
losing. Therefore, being able to use a predefined playbook that would allow a team to predict how best to 
set the field based on the context of the game, the batsman they are bowling to and the bowlers on hand 
would give them a significant strategic advantage. 

## BACKGROUND ON CRICKET

Cricket is a sport played by two teams with each side having eleven players. Each team is a right blend of 
batsmen, bowlers and allrounders. The batsmen’s role is to score maximum runs possible and the bowlers 
have to take maximum wickets and restrict the other team from scoring runs at thes ame time. All-rounders 
are the players who can both bat and bowl and they contribute by scoring uns and taking wickets.  
The aim of cricket is simple: score more runs than the other team. Scoring runs is conducted, with one player 
bowling the ball to a batsman who defends 3 wooden stumps (called wickets) and attempts to hit the ball in 
order to try and accumulate runs. The legal hitting area in cricket is 360 degrees from the location of the 
batsman, who plays on a rectangular pitch in the centre of the playing field as demonstrated in Figure 1.  
![image](https://github.com/user-attachments/assets/914cf583-fe13-41f5-ac70-d839db697d37)



Figure. 1 Cricket Pitch 
This large hitting area must be covered by 10 fielders, one of which is the wicketkeeper  who typically 
stands directly behind the batsman. Scoring runs can done in one of two ways. Firstly, by hitting boundaries, 
4 runs if the ball clears the playing area along the ground and 6 runs, if it's hit over the playing area. Secondly - and most frequently - a batsman can score runs (1, 2 or 3) by swapping with their partner who stands at 
the opposite end of the pitch before the ball is returned to the stumps. Once 10 of a team’s 11 batsmen have 
been dismissed their innings is complete, even if there are balls left to bowl. The goal of the fielding team 
is to dismiss the batsman and/or limit their run-scoring. 


## OBJECTIVES

The following are the objectives of this project: 
1. Predict Effectiveness of shot played by batsman- based on ball type, bowler, match conditions and 
historical data. 

2. Employ feature engineering to derive statistically significant metrics 

3. Identify metrics that exert substantial influence on the prediction process.


## Ball-by-Ball Data Acquisition 
###  DATA COLLECTION 
The dataset is a set of India matches from the ODI World Cup held in 2019 and has been built by web 
scraping ball by ball data from ESPN Cric Info. Each record in the dataset represents a ball in an over and 
contains details of each ball and the batsman who faces it along with the type of shot played by him. Each 
player is given a unique player ID. Our dataset contains the usual features like batsman and bowler name, 
over number, ball number, runs scored, etc. Apart from the usual features, there are ‘Control’, ‘Hit the Bat’ 
and ‘Effectiveness’ features. The ‘Control’ feature puts a binary value (0 or 1) to how much control the 
batsman has on the shot, in terms of timing and placement. ‘Hit the bat’ is a simple binary feature 
representing contact between bat and ball. ‘Effectiveness’ feature is a number that rates a shot's effectiveness 
against a particular ball type in the given conditions, ranging from pitch type to over number, bowler and 
ball type.

## DATA PREPROCESSING  

A large number of values in the Wicket, Dismissal kind and Caught/Bowled by columns were null. This is 
because there are 10 wickets that can be taken in an innings and the number of balls bowled is usually much 
beyond 10. We treated such missing values by replacing them with zero. Columns with binary values such as 
‘Wicket’ (where value can be either Wicket or No Wicket), were binary encoded into integers (0 or 1). 
Categorical variables were encoded using Label Encoder, after which all the feature variables were 
brought to scale using Standard Scaler.

## EXPLORATORY DATA ANALYSIS 

For a comprehensive understanding of the correlation between the features and our target variable 
(Effectiveness), the seaborn library was employed to generate a heatmap. The heatmap visually 
represented the connections between each mean statistic and the ‘Effectiveness’, utilizing data from every 
ball in the data frame. 



![image](https://github.com/user-attachments/assets/280ed19d-f75b-4fd3-ba89-976e89a8fc04)




![image](https://github.com/user-attachments/assets/99045d18-fd35-41be-b931-a93af18c7f53)

## FEATURE ENGINEERING 

Formally, feature engineering has been employed to systematically generate additional features beyond the 
basic ones available on the website. This deliberate augmentation aims to bolster the accuracy of the 
prediction model by incorporating various statistics often overlooked in news channels and common 
discussions. The strategic creation of these features enhances the model's ability to capture nuanced patterns 
and contribute to more informed and precise predictions. These features are calculated based on the analysis 
parameters calculated.

![image](https://github.com/user-attachments/assets/ec69bdcd-a6ad-4034-9ceb-83acff188077)

## IMPORTANT FEATURES

![image](https://github.com/user-attachments/assets/25b6ac11-0c3a-4844-8425-8c04cadd6c22)


# Machine Learning Model  
## TARGET VARIABLE 
Our target variable is a continuous variable of float type ‘Effectiveness’, hence making this a regression 
problem. After testing the association of categorical variables with the target variable using a chi square test 
and encoding them after elimination, we move on to splitting our dataset into training and testing data after 
scaling the features.  


## LEARNING ALGORITHMS 
For generating the prediction models, we used supervised machine learning algorithms. In supervised 
learning algorithms, each training tuple is labelled with its corresponding target variable(Effectiveness). 
After a process of elimination using mean absolute error (MAE) as a metric during cross validation, we boil 
down to the Random Forest, Decision Tree and XGBoost Regressors, as these had the least MAE. We used 
Decision Tree Regression, Random Forest regression and XGBoost Regression for our experiments. These 
algorithms are explained in brief.



![image](https://github.com/user-attachments/assets/bd1d7081-aecc-4298-bf92-d51ecb61d9e0)


## DECISION TREE REGRESSOR 

This regression model consists of an ensemble of decision trees. Each tree in a regression decision forest 
outputs a Gaussian distribution as a prediction. An aggregation is performed over the ensemble of trees to 
find a Gaussian distribution closest to the combined distribution for all trees in the model. Decision Trees 
are able to handle missing values and outliers in the data much better as it is not affected by outliers because 
it splits the data based on the feature values. 


## RANDOM FOREST REGRESSOR 

Random forest regression is a supervised learning algorithm and bagging technique that uses an ensemble 
learning method for regression in machine learning. The trees in random forests run in parallel, meaning 
there is no interaction between these trees while building the trees. A random forest is a meta estimator that 
fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to 
improve the predictive accuracy and control over-fitting. 

## XGBOOST REGRESSOR 

Gradient boosting refers to a class of ensemble machine learning algorithms that can be used for 
classification or regression predictive modeling problems. Ensembles are constructed from decision tree 
models. Trees are added one at a time to the ensemble and fit to correct the prediction errors made by prior 
models. This is a type of ensemble machine learning model referred to as boosting. XGBoost is an efficient 
implementation of gradient boosting that can be used for regression predictive modeling.  

## EVALUATION METRIC 
R2 Score:  
An R-Squared value shows how well the model predicts the outcome of the dependent variable. R-Squared 
values range from 0 to 1. An R-Squared value of 0 means that the model explains or predicts 0% of the 
relationship between the dependent and independent variables. R2 is useful when the goal is to explain the 
variability in the target variable using the predictors. 


## HYPERPARAMETER TUNING 
Speaking in statistical terms, hyperparameter tuning captures a snapshot of the current performance of a 
model, and compares this snapshot with others taken previously. In any machine learning algorithm, 
hyperparameters need to be initialized before a model starts the training. Fine-tuning the model 
hyperparameters maximizes the performance of the model on a validation set. In a machine learning 
context, a hyperparameter is a parameter whose value is set before initiating the learning process. On the 
other hand, the values of model parameters are derived via training the data. Model parameters refer to the 
weights and coefficients, which are derived from the data by the algorithm. Every algorithm has a defined 
set of hyperparameters; for example, for a Decision Tree, this is a depth parameter.


![image](https://github.com/user-attachments/assets/752c1e8e-8861-41d4-a4d3-b5ef68d3528b)


# Results 
## PREDICTIONS 

The XGBoost regressor model had the highest R2 Score by an inch with a value of 0.918, followed by 
Random Forest regressor with a R2 score of 0.917 and then the Decision Tree regressor with a R2 score of 
0.885.  

![image](https://github.com/user-attachments/assets/abcbdf63-fac3-4d65-89ee-07c9fae18a86)

## MOST IMPACTFUL STATISTICS 
The Regressor will predict a higher shot effectiveness if:  

● Number of run scores against that particular ball is high. 

● The Control on the shot is higher, meaning the shot intended to be played by the batsman matched the 
end result of the shot more often than not. 

● Hit the Bat has a value of 1, meaning the batsman made connection with the ball, immediately 
increasing the chances of high shot effectiveness.

![image](https://github.com/user-attachments/assets/97f5aa87-248e-42b5-9841-de6a1d697fb3)


## DISCUSSION 

The three main factors that the fielding team can control to increase the probability of a favorable outcome 
are: 


i.The placement of fielders (subject to restrictions on the number of players patrolling specific 
regions)  

ii.The choice of bowler at a given stage of a match (subject to a maximum number of balls per 
12 
bowler)  

iii. 
The speed and trajectory of the ball (subject to the skill and consistency of the bowler)  
What the fielding team cannot control at the moment of the delivery is the match context, 
atmospheric/weather conditions and factors intrinsic to the batsman, such as their level of skill or decision
making. However, the fielding team can leverage their understanding of the ability and tendencies of a 
batsman to restrict their impact. 
Our model helps understand these tendencies of a batsman by leveraging historical data and training the 
model to predict a batsman’s shot effectiveness depending on the batsman’s success rate against a bowler 
and ball type, and the ball type and bowler themselves in the match context (over number, wickets fallen, 
etc.)


## CONCLUSION  

In this project, we have proposed a machine learning model, which utilizes our shot type, ball type 
and effectiveness definitions to predict likely batsman shot effectiveness, given ball type and 
match situation. The power of our predictions is that rather than relying on either broad averages 
that ignore context or ever-diminishing sample sizes, we can isolate and account for match context, 
the sequential nature of cricket and the individual tendencies of batsmen and bowlers. As a result, 
our predictions can add value to both pre-match line-up decisions that are most suitable to the 
opposition and venue, as well as in-game tactics. 
Ultimately, cricket is a sport often decided by the barest of margins: just a single run saved by New 
Zealand across their 300 deliveries would have been enough to see them win the World Cup. Our 
predictions can provide vital information to inform the decision-making of coaches and captains, 
both in terms of pre-match and in-game tactical choices. 

## FUTURE WORK 

The model can be extended to help bowlers search for a wicket as well. Often in a test match, the 
process of creating a wicket starts a long time before the wicket taking delivery is bowled, with 
the bowler setting up the batsman for a few balls to overs before the fatal blow. Creating bowling 
plans for a specific batsman in specific conditions is a possibility.  
Including the ball release angle, line of the ball, impact area on the bat, temperature and humidity 
on match day as features can provide more accurate shot effectiveness predictions. The problem 
rises in scraping these data as these data are not available online for free scraping and only 
companies working along with cricket teams, cricket councils like the ICC (International Cricket 
Council) and leagues such as IPL collect such data and they are not made available to the public. 
Even scraping the ball-by-ball data we collected had to be done so from ESPN Cric Info 
commentary of matches and was never available ready to use.  Thus, there is definitely scope for 
improvement in the future but, it depends a lot on availability of data. 

