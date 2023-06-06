import pandas as pd
from   sklearn.linear_model    import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics           import roc_curve
from sklearn.metrics           import roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.datasets import make_classification

from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import AdaBoostClassifier
from sklearn.ensemble        import RandomForestClassifier

from   sklearn               import metrics
from imblearn.over_sampling import SMOTE

# Filtering Warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
import warnings
warnings.filterwarnings("ignore")

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

PATH     = "/Users/flaca/PycharmProjects/COMP4254/Datasets/"

# Load Data
CSV_DATA = "diabetes_binary_health_indicators_BRFSS2015.csv"
df  = pd.read_csv(PATH + CSV_DATA)
df.columns = ['Diabetes_binary', 'High Blood Pressure', 'High Cholesterol', 'Cholesterol Checked', 'BMI',
              'Smoker', 'Stroke', 'Heart Attack Diagnosed', 'Physical Activity', 'Fruits', 'Veggies',
              'Heavy Alcohol Consumption', 'Any Health Care', 'No Doctor Cost', 'General Health',
              'Mental Health', 'Physical Health', 'Difficulty Walking', 'Sex', 'Age', 'Education', 'Income']

print(df.head())
print(df.describe())
print(df.columns)
print(df.isna().sum())

# Adding bins
df['BmiBin']   = pd.cut(x=df['BMI'], bins=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
df['PhysHlthBin']   = pd.cut(x=df['Physical Health'], bins=[0, 5, 10, 15, 20, 25, 30])
binCols = ['BmiBin', 'PhysHlthBin']
dfBins = df[binCols]
dummyDf = pd.get_dummies(dfBins, columns=binCols)
df      = pd.concat(([df, dummyDf]), axis=1)

# Renaming bin variables
df.rename(columns={"BmiBin_(10, 20]": "BMI (10-20)"}, inplace=True)
df.rename(columns={"BmiBin_(20, 30]": "BMI (20-30)"}, inplace=True)
df.rename(columns={"BmiBin_(30, 40]": "BMI (30-40)"}, inplace=True)
df.rename(columns={"BmiBin_(40, 50]": "BMI (40-50)"}, inplace=True)
df.rename(columns={"BmiBin_(50, 60]": "BMI (50-60)"}, inplace=True)
df.rename(columns={"BmiBin_(60, 70]": "BMI (60-70)"}, inplace=True)
df.rename(columns={"BmiBin_(70, 80]": "BMI (70-80)"}, inplace=True)
df.rename(columns={"BmiBin_(80, 90]": "BMI (80-90)"}, inplace=True)
df.rename(columns={"BmiBin_(90, 100]": "BMI (90-100)"}, inplace=True)

df.rename(columns={"PhysHlthBin_(0, 5]": "Physical Health (0-5)"}, inplace=True)
df.rename(columns={"PhysHlthBin_(5, 10]": "Physical Health (5-10)"}, inplace=True)
df.rename(columns={"PhysHlthBin_(10, 15]": "Physical Health (10-15)"}, inplace=True)
df.rename(columns={"PhysHlthBin_(15, 20]": "Physical Health (15-20)"}, inplace=True)
df.rename(columns={"PhysHlthBin_(20, 25]": "Physical Health (20-25)"}, inplace=True)
df.rename(columns={"PhysHlthBin_(25, 30]": "Physical Health (25-30)"}, inplace=True)
print(df.columns)

# Seperate the target and independent variable
X = df.copy()     # Create separate copy to prevent unwanted tampering of data.
del X['Diabetes_binary']  # Delete target variable.
del X['BmiBin']
del X['PhysHlthBin']
del X['BMI']
del X['Physical Health']

# Target variable
y = df['Diabetes_binary']

feature_list = list(X.columns)
print(feature_list)

### Selecting Important Features ### -----------------------------------------------------
### Chi-Square Test ### ------------------------------------------------------------------
test = SelectKBest(score_func=chi2, k=4)
chiScores = test.fit(X, y) # Summarize scores
np.set_printoptions(precision=3)

# Search here for insignificant features.
print("\nPredictor Chi-Square Scores: " + str(chiScores.scores_))

# Create a sorted list of the top features.
dfFeatures = pd.DataFrame()
for i in range(0, len(chiScores.scores_)):
    featureObject = {"feature":feature_list[i], "chi-square score":chiScores.scores_[i]}
    dfFeatures    = dfFeatures.append(featureObject, ignore_index=True)

print("\n*** Chi-Square Test ***")
dfFeatures = dfFeatures.sort_values(by=['chi-square score'], ascending=False, ignore_index=True)
print(dfFeatures.head(5))

chiFeatures = list(dfFeatures['feature'])[:4]

# Plot Top Features
chi_importances = pd.Series(list(dfFeatures['chi-square score'])[:10], index=dfFeatures['feature'][:10])
std = np.std(dfFeatures['chi-square score'][:10])

fig, ax = plt.subplots()
chi_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Top Features with Chi-Square Test")
ax.set_ylabel("Mean decrease in impurity")
plt.xticks(rotation=45, ha='right')
fig.tight_layout()
ax.set(xlabel=None)
plt.show()

### Forward Feature Elimination ### ------------------------------------------------------
#  f_regression is a scoring function to be used in a feature selection procedure
#  f_regression will compute the correlation between each regressor and the target
ffs = f_regression(X, y)

dfFeatures = pd.DataFrame()
for i in range(0, len(X.columns) - 1):
    featureObject = {"feature":feature_list[i], "ffs score":ffs[0][i]}
    dfFeatures    = dfFeatures.append(featureObject, ignore_index=True)

print("\n*** Forward feature elimination ***")
dfFeatures = dfFeatures.sort_values(by=['ffs score'], ascending=False, ignore_index=True)
print(dfFeatures.head(5))

ffsFeatures = list(dfFeatures['feature'])[:4]

# Plot Top Features
ffs_importances = pd.Series(list(dfFeatures['ffs score'])[:10], index=dfFeatures['feature'][:10])
std = np.std(dfFeatures['ffs score'][:10])

fig, ax = plt.subplots()
ffs_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Top Features with Forward Feature Elimination")
ax.set_ylabel("Mean decrease in impurity")
plt.xticks(rotation=45, ha='right')
ax.set(xlabel=None)
fig.tight_layout()
plt.show()

#### Recursive Feature Elimination ### ------------------------------------------------------
# Create the object of the model
model = LogisticRegression(solver='liblinear')

# Specify the number of  features to select
rfe = RFE(estimator=model, n_features_to_select=5)

# fit the model
rfe = rfe.fit(X, y)

print("\n*** Recursive Feature Elimination ***")
rfeFeatures = []
for i in range(0, len(X.keys())):
    if(rfe.support_[i]):
        print(X.keys()[i])
        rfeFeatures.append(X.keys()[i])

### Feature Importance ### ----------------------------------------------------------------
FEATURES = df.copy()     # Create separate copy to prevent unwanted tampering of data.
del FEATURES['Diabetes_binary']  # Delete target variable.
del FEATURES['BmiBin']
del FEATURES['PhysHlthBin']
del FEATURES['BMI']
del FEATURES['Physical Health']
FEATURES = np.array(FEATURES)
labels = np.array(df['Diabetes_binary'])

FEATURES, labels = make_classification(
    n_samples=10000,
    n_features=10,
    n_informative=5,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    random_state=0,
    shuffle=False,
)
X_train, X_test, y_train, y_test = train_test_split(FEATURES, labels, test_size = 0.3, random_state=42)

feature_names = [f"{feature_list[i]}" for i in range(FEATURES.shape[1])]
forest = RandomForestClassifier(random_state=0)
forest.fit(X_train, y_train)
y_pred=forest.predict(X_test)
print("\n*** Feature Importance ***")
print("Accuracy 1:", metrics.accuracy_score(y_test, y_pred))

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

# Plot Top Features: Mean Decrease Impurity
forest_importances = pd.Series(importances, index=feature_names)
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
plt.xticks(rotation=45, ha='right')
fig.tight_layout()
plt.show()

# Display Feature Importance
# Get numerical feature importances
importances = list(forest.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_names, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

# selected features
impForest = ['High Blood Pressure','Cholesterol Checked','Heavy Alcohol Consumption',
             'BMI (10-20)', 'BMI (20-30)']

### Functions ### -----------------------------------------------------------------------------
def getUnfitModels():
    models = list()
    models.append(LogisticRegression())
    models.append(DecisionTreeClassifier())
    models.append(AdaBoostClassifier())
    models.append(RandomForestClassifier(n_estimators=10))
    return models


def evaluateModel(y_test, predictions, model):
    precision = round(precision_score(y_test, predictions), 4)
    recall = round(recall_score(y_test, predictions), 4)
    f1 = round(f1_score(y_test, predictions), 4)
    accuracy = round(accuracy_score(y_test, predictions), 4)

    print("Precision:" + str(precision) + " Recall:" + str(recall) + \
          " F1:" + str(f1) + " Accuracy:" + str(accuracy) + \
          "   " + model.__class__.__name__)


def fitBaseModels(X_train, y_train, X_test, y_test, models):
    dfPredictions = pd.DataFrame()

    # Fit base model and store its predictions in dataframe.
    for i in range(0, len(models)):
        models[i].fit(X_train, y_train)
        predictions = models[i].predict(X_test)
        evaluateModel(y_test, predictions, models[i])
        colName = str(i)
        # Add base model predictions to column of data frame.
        dfPredictions[colName] = predictions
    return dfPredictions, models


def fitStackedModel(X, y):
    X_train_SMOTE, y_train_SMOTE = SMOTE().fit_resample(X, y)
    model = LogisticRegression(solver='newton-cholesky', max_iter=1000, fit_intercept=True, random_state=0)
    model.fit(X_train_SMOTE, y_train_SMOTE)
    return model


def fitAllModels(X, y):
    # Get base models.
    unfitModels = getUnfitModels()
    models = None
    stackedModel = None
    kfold = KFold(n_splits=3, shuffle=True)
    y = y.to_frame()
    numFold = 1
    for train_index, test_index in kfold.split(X):
        X_train = X.loc[X.index.intersection(train_index), :]
        X_test = X.loc[X.index.intersection(test_index), :]
        y_train = y.loc[y.index.intersection(train_index), :]
        y_test = y.loc[y.index.intersection(test_index), :]

        # Fit base and stacked models.
        print("*** K-fold:", numFold, "***")
        dfPredictions, models = fitBaseModels(X_train, y_train, X_test, y_test, unfitModels)
        stackedModel = fitStackedModel(dfPredictions, y_test)
        numFold += 1
    return models, stackedModel


def evaluateBaseAndStackModelsWithUnseenData(X, y, models, stackedModel, currentDataset):
    # Evaluate base models with validation data.
    print("\n** Evaluate Base Models **")
    dfValidationPredictions = pd.DataFrame()
    for i in range(0, len(models)):
        predictions = models[i].predict(X)
        y_prob = models[i].predict_proba(X)
        colName = str(i)
        dfValidationPredictions[colName] = predictions
        evaluateModel(y, predictions, models[i])

        auc = roc_auc_score(y_test, y_prob[:, 1], )
        print('Logistic: ROC AUC=%.3f' % (auc))

    # Evaluate stacked model with validation data.
    stackedPredictions = stackedModel.predict(dfValidationPredictions)
    print("\n*** Evaluate Stacked Model:", currentDataset, " ***")
    evaluateModel(y, stackedPredictions, stackedModel)

    # Calculate ROC AUC Score
    stackedProbs = stackedModel.predict_proba(dfValidationPredictions)
    stackedAUC = roc_auc_score(y, stackedProbs[:, 1], )
    print("Stacked AUC: ", stackedAUC)

    # Plot ROC
    lr_fpr, lr_tpr, _ = roc_curve(y, stackedProbs[:, 1])
    plt.plot(lr_fpr, lr_tpr, marker='.', label=currentDataset)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot([0, 1], [0, 1], '--', label='No Skill')
    plt.legend()
    plt.show()

### Set feature sets and data scaling ### ----------------------------------------------------
# Selected Important Features by Chi-Square, FFS, RFE, and Feature Importance
featureSet = [chiFeatures, ffsFeatures, rfeFeatures, impForest]
methods = ['Chi-Square Test', 'Forward Feature Elimination', 'Recursive Feature Elimination', 'Feature Importance']

# Scaling
MinMax_x    = MinMaxScaler()
SD_x        = StandardScaler()
Robust_x    = RobustScaler()
SCALES = [MinMax_x, SD_x, Robust_x]
scMethods = ['MinMax Scaler', 'Standard Scaler', 'Robust Scaler']

# SMOTE
smt = SMOTE()

### Testing and Evaluating Models with Scaling Data ### -------------------------------------------------
for i, Scaler in enumerate(SCALES):
    print("***********************")
    print("\n***", scMethods[i], "***\n")
    print("***********************")
    for idx, features in enumerate(featureSet):
        currentFeatures = methods[idx]
        # Re-assign X with significant columns only after chi-square test.
        X_subset = X[features]
        X_Scaled = Scaler.fit_transform(X_subset)

        features = list(X_subset.keys())
        dfXScaledWithHeaders = pd.DataFrame(data=X_Scaled, columns=features)

        # Split data.
        X_train, X_test, y_train, y_test = train_test_split(dfXScaledWithHeaders, y, train_size=0.7)

        print("\n**** FITTING MODELS: ", currentFeatures)
        models, stackedModel = fitAllModels(X_train, y_train)

        print("\n**** Evaluating models with unseen data: ")
        evaluateBaseAndStackModelsWithUnseenData(X_test, y_test, models, stackedModel, currentFeatures)


### Final Three Models
### Build Models and Evaluate Models with Scaling ### -------------------------------------------------
modelA = ['High Blood Pressure','General Health','Difficulty Walking', 'Age']
modelB = ['High Blood Pressure','General Health','Difficulty Walking', 'Age', 'High Cholesterol']
modelC = ['High Blood Pressure','General Health','Difficulty Walking', 'Age', 'Heart Attack Diagnosed']
methods = ['Model A','Model B','Model C']

featureSet = [modelA, modelB, modelC]

for i, Scaler in enumerate(SCALES):
    print("***********************")
    print("\n***", scMethods[i], "***\n")
    print("***********************")
    for idx, features in enumerate(featureSet):
        currentFeatures = methods[idx]
        # Re-assign X with significant columns only after chi-square test.
        X_subset = X[features]
        X_Scaled = Scaler.fit_transform(X_subset)

        features = list(X_subset.keys())
        dfXScaledWithHeaders = pd.DataFrame(data=X_Scaled, columns=features)

        # Split data.
        X_train, X_test, y_train, y_test = train_test_split(dfXScaledWithHeaders, y, train_size=0.7)

        print("\n**** FITTING MODELS: ", currentFeatures)
        models, stackedModel = fitAllModels(X_train, y_train)

        print("\n**** Evaluating models with unseen data: ")
        evaluateBaseAndStackModelsWithUnseenData(X_test, y_test, models, stackedModel, currentFeatures)