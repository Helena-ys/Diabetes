# Predictive Modeling: Diabetes

This report utilizes a cleaned and consolidated dataset from the BRFSS 2015 dataset that is available on Kaggle. Several data transformations, analyses, tests, and comparisons were conducted to investigate the predictors of diabetes health risks using BRFSS’s Annual Survey data in 2015. The findings revealed that three components of the diabetes health indicator dataset significantly influenced the prediction of an individual's likelihood of having diabetes. These components include General Health, High Blood Pressure, and Difficulty Walking.

### Exploratory Analysis
- Impact of General Health, Difficulty Walking, and High Blood Pressure on the prediction of diabetes.
  - General Health: The general health column is divided into five categories, with 1 indicating excellent, 2 indicating very good, 3 indicating good, 4 indicating fair, and 5 indicating poor. Individuals with a poor general health condition may have an increased risk of developing diabetes, up to 35%.
    ![GeneralHealth](https://github.com/Helena-ys/Diabetes/blob/main/Chart_General%20Health.jpg?raw=true)
    <img src="[https://user-images.githubusercontent.com/16319829/81180309-2b51f000-8fee-11ea-8a78-ddfe8c3412a7.png](https://github.com/Helena-ys/Diabetes/blob/main/Chart_General%20Health.jpg)" width=50% height=50%>
  - High Blood Pressure: Individuals who have experienced high blood pressure may have up to an 18% increased risk of developing diabetes. It is worth noting that approximately 75% of individuals who have experienced high blood pressure also have diabetes.
    ![HighBP](https://github.com/Helena-ys/Diabetes/blob/main/Chart_HighBP.jpg?raw=true)
  - Difficulty Walking: Compared to individuals who do not experience difficulty when walking or climbing stairs, those who do have a 20% higher risk of developing diabetes.
    ![DiffWalking](https://github.com/Helena-ys/Diabetes/blob/main/Chart_DiffWalking.jpg?raw=true)

Data Source: Diabetes Health Indicators Dataset 
https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
