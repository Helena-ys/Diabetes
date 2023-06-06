# Predictive Modeling: Diabetes

This report utilizes a cleaned and consolidated dataset from the BRFSS 2015 dataset that is available on Kaggle. Several data transformations, analyses, tests, and comparisons were conducted to investigate the predictors of diabetes health risks using BRFSS’s Annual Survey data in 2015. The findings revealed that three components of the diabetes health indicator dataset significantly influenced the prediction of an individual's likelihood of having diabetes. These components include General Health, High Blood Pressure, and Difficulty Walking.

## Exploratory Analysis
### High Target Predictors
- Impact of General Health, Difficulty Walking, and High Blood Pressure on the prediction of diabetes.
  - **General Health**: The general health column is divided into five categories, with 1 indicating excellent, 2 indicating very good, 3 indicating good, 4 indicating fair, and 5 indicating poor. Individuals with a poor general health condition may have an increased risk of developing diabetes, up to 35%.
    
    <img src="https://github.com/Helena-ys/Diabetes/blob/main/Chart_General%20Health.jpg" width=50% height=50%>
  - **High Blood Pressure**: Individuals who have experienced high blood pressure may have up to an 18% increased risk of developing diabetes. It is worth noting that approximately 75% of individuals who have experienced high blood pressure also have diabetes.

    <img src="https://github.com/Helena-ys/Diabetes/blob/main/Chart_HighBP.jpg" width=50% height=50%>
  - **Difficulty Walking**: Compared to individuals who do not experience difficulty when walking or climbing stairs, those who do have a 20% higher risk of developing diabetes.
    
    <img src="https://github.com/Helena-ys/Diabetes/blob/main/Chart_DiffWalking.jpg" width=50% height=50%>

## Data Preparation
### Data Transformation Techniques
- Filling in missing data
- Converting string data to numeric data
- Splitting continuous numerical data into bins

## Feature Selection
To identify significant features in a dataset, the following algorithms used:
- Chi-square Test
- Forward Feature Elimination
- Recursive Feature Elimination 
- Feature Importance

### Chi-square Test
The chart below highlights the 5 most significant features: "Mental Health", "Difficulty Walking", "High Blood Pressure", "General Health", and “Age”.

### Forward Feature Elimination
The chart below highlights the 4 most significant features: "General Health", "High Blood Pressure", "Difficulty Walking", and "High Cholesterol”.

### Recursive Feature Elimination
RFE is an iterative method that involves training a model on all features, ranking the features based on their importance, and recursively removing the least important features until the desired number of features is achieved. By this algorithm, the following five significant variables were selected as significant: "High Blood Pressure", "Cholesterol Checked", "Heavy Alcohol Consumption", "BMI_(10-20)", and "BMI_(20-30)". 

### Feature Importance
Random forests have a feature that allows them to rank variables in order of importance or significance for predicting the target variable. This feature is based on the Gini importance or mean decrease impurity. The chart below highlights the 5 most significant features: "High Cholesterol", “Cholesterol Checked”, “Stroke”, "High Blood Pressure", and “Smoker”.

## Model Evaluation
During the process of evaluating models, the following techniques were utilized to enhance the accuracy and robustness of the model.
- Scaling methods: Scaling was used to normalize the data and improve its performance.
- K-fold cross validation
- Over-sampling
- Stacking classifiers

### Stacking Classifier Model
Stacking classifier is a machine learning ensemble method that involves combining multiple individual classifiers to improve the overall predictive performance. 
In this analysis, the following classifiers combined: 
- Logistic regression
- Decision Tree Classifier
- AdaBoost Classifier
- Random Forest Classifier

### Over-sampling technique
To improve the performance of classifiers on imbalanced datasets.
- SMOTE: Synthetic Minority Over-sampling Technique

### Model Comparison Matrix
<img src="https://github.com/Helena-ys/Diabetes/blob/main/Model_Comparison_Matrix.JPG" width=60% height=60%>

## Conclusion
Based on the Model Comparison Matrix presented above, the combination of high blood pressure, general health, blood cholesterol, age group, and difficulty walking were found to be the most accurate predictors for an individual's diabetes.

During the multiple tests conducted on various combinations of significant variables, the features selected by RFE and Feature Importance algorithms generated an "Undefined Metric Warning". To address this issue, the "zero_division" parameter can be implemented to control this behavior. Implementing this parameter could potentially improve the model and help to develop a more accurate model.


### Data Source
Diabetes Health Indicators Dataset 
https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
