import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import os, pathlib

#https://towardsdatascience.com/feature-selection-correlation-and-p-value-da8921bfb3cf

np.random.seed(123)

cur_dir = os.getcwd()
input = f"{cur_dir}/data/cancer_data.csv"
data = pd.read_csv(input)
data = data.iloc[:,1:-1]
label_encoder = LabelEncoder()
data.iloc[:,0] = label_encoder.fit_transform(data.iloc[:,0]).astype('float64')

corr = data.corr()
sns.heatmap(corr)

columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = data.columns[columns]
data = data[selected_columns]


#Selecting columns based on p-value
selected_columns = selected_columns[1:].values
import statsmodels.api as sm #statsmodels.formula.api as sm


def backwardElimination(x, Y, sl, columns):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    columns = np.delete(columns, j)

    regressor_OLS.summary()
    return x, columns
SL = 0.05
data_modeled, selected_columns = backwardElimination(data.iloc[:, 1:].values, data.iloc[:, 0].values, SL,
                                                     selected_columns)


result = pd.DataFrame()
result['diagnosis'] = data.iloc[:,0]
data = pd.DataFrame(data = data_modeled, columns = selected_columns)

#Visualizing the selected features
fig = plt.figure(figsize = (20, 25))
j = 0
for i in data.columns:
    plt.subplot(6, 4, j+1)
    j += 1
    sns.distplot(data[i][result['diagnosis']==0], color='g', label = 'benign')
    sns.distplot(data[i][result['diagnosis']==1], color='r', label = 'malignant')
    plt.legend(loc='best')
fig.suptitle('Breast Cancer Data Analysis')
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()

x_train, x_test, y_train, y_test = train_test_split(data.values, result.values, test_size = 0.2)
#Building a model with the selected features
svc=SVC() # The default kernel used by SVC is the gaussian kernel
svc.fit(x_train, y_train)
prediction = svc.predict(x_test)
#We are using a confusion matrix here
cm = confusion_matrix(y_test, prediction)
sum = 0
for i in range(cm.shape[0]):
    sum += cm[i][i]
accuracy = sum / x_test.shape[0]
print(accuracy)

##########
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from warnings import filterwarnings
filterwarnings("ignore")

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
cancer = pd.read_csv(input)
cancer = cancer.drop('id', axis=1)
cancer.sample(5)


def EDA(df):
    print('\033[1m' + 'Shape of the data :' + '\033[0m')
    print(df.shape,
          '\n------------------------------------------------------------------------------------\n')

    print('\033[1m' + 'All columns from the dataframe :' + '\033[0m')
    print(df.columns,
          '\n------------------------------------------------------------------------------------\n')

    print('\033[1m' + 'Datatpes and Missing values:' + '\033[0m')
    print(df.info(),
          '\n------------------------------------------------------------------------------------\n')

    print('\033[1m' + 'Summary statistics for the data' + '\033[0m')
    print(df.describe(include='all'),
          '\n------------------------------------------------------------------------------------\n')

    print('\033[1m' + 'Outliers in the data :' + '\033[0m')
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
    print(outliers.sum(),
          '\n------------------------------------------------------------------------------------\n')

    print('\033[1m' + 'Memory used by the data :' + '\033[0m')
    print(df.memory_usage(),
          '\n------------------------------------------------------------------------------------\n')

    print('\033[1m' + 'Number of duplicate values :' + '\033[0m')
    print(df.duplicated().sum())

#EDA(cancer)

# Dropping Unnamed column
cancer = cancer.loc[:, ~cancer.columns.str.contains('^Unnamed')]

# Encoding target variable
cancer.diagnosis = cancer.diagnosis.astype('category')
cancer.diagnosis = cancer.diagnosis.cat.codes
cancer.diagnosis.value_counts()

cancer_mean = cancer.loc[:, 'radius_mean':'fractal_dimension_mean']
cancer_mean['diagnosis'] = cancer['diagnosis']
# Plotly's Scatterplot matrix

dimensions = []
for col in cancer_mean:
    dimensions.append(dict(label=col, values=cancer_mean[col]))

fig = go.Figure(data=go.Splom(
    dimensions=dimensions[:-2],
    showupperhalf=False,
    diagonal_visible=False,
    marker=dict(
        color='rgba(135, 206, 250, 0.5)',
        size=5,
        line=dict(
            color='MediumPurple',
            width=0.5))
))

fig.update_layout(
    title='Pairplot for mean attributes of the dataset',
    width=1100,
    height=1500,
)

fig.show()


# Correlation matrix

plt.figure(figsize = (20, 12), dpi = 150)

corr = cancer.corr()
mask = np.triu(np.ones_like(corr, dtype = bool))

sns.heatmap(corr,
            mask = mask,
            cmap = 'BuPu',
            annot = True,
            linewidths = 0.5,
            fmt = ".2f")

plt.title('Correlation Matrix',
          fontsize = 20,
          weight = 'semibold',
          color = '#de573c')
plt.show()


def subplot_titles(cols):
    '''
    Creates titles for the subplot's subplot_titles parameter.
    '''
    titles = []
    for i in cols:
        titles.append(i + ' : Distribution')
        titles.append(i + ' : Violin plot')
        titles.append(i + ' by Diagnosis')

    return titles


def subplot(cols, row=0, col=3):
    '''
    Takes a dataframe as an input and returns distribution plots for each variable.
    '''
    row = len(cols)
    fig = make_subplots(rows=row, cols=3, subplot_titles=subplot_titles(cols))

    for i in range(row):
        fig.add_trace(go.Histogram(x=cancer[cols[i]],
                                   opacity=0.7),
                      row=i + 1, col=1)

        fig.add_trace(go.Violin(y=cancer[cols[i]],
                                box_visible=True),
                      row=i + 1, col=2)

        fig.add_trace(go.Box(
            y=cancer[cols[i]][cancer.diagnosis == 0],
            marker_color='#6ce366',
            name='Benign'
        ), row=i + 1, col=3)

        fig.add_trace(go.Box(
            y=cancer[cols[i]][cancer.diagnosis == 1],
            marker_color='#de5147',
            name='Malignant'
        ), row=i + 1, col=3)

    for i in range(row):
        fig.update_xaxes(title_text=cols[i], row=i + 1)

    fig.update_yaxes(title_text="Count")
    fig.update_layout(height=450 * row, width=1100,
                      title='Summary of mean tumor attributes (For Diagnois : Green=Benign, Red=Malignant)',
                      showlegend=False,
                      plot_bgcolor="#f7f1cb"
                      )

    fig.show()


x = subplot(cancer.drop('diagnosis', axis=1).columns)


def outlier(df):
    df_ = df.copy()
    df = df.drop('diagnosis', axis=1)

    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)

    iqr = q3 - q1

    lower_limit = q1 - (1.5 * iqr)
    upper_limit = q3 + (1.5 * iqr)

    for col in df.columns:
        for i in range(0, len(df[col])):
            if df[col][i] < lower_limit[col]:
                df[col][i] = lower_limit[col]

            if df[col][i] > upper_limit[col]:
                df[col][i] = upper_limit[col]

    for col in df.columns:
        df_[col] = df[col]

    return (df_)


cancer = outlier(cancer)
#https://www.kaggle.com/toomuchsauce/breast-cancer-prediction-plotly-99-12-val-acc
#We can use several techniques (Chi-sq test, Random Forest Importance, Forest Feature Selection, Exhaustive Feature Selection, fisher score just to name a few)
#but here, I'll use Variance inflation factor.
#Feature Selection using VIF
#
def VIF(df):
    vif = pd.DataFrame()
    vif['Predictor'] = df.columns
    vif['VIF'] = [variance_inflation_factor(df.values, col) for col in range(len(df.columns))]
    return vif


X = cancer.drop('diagnosis', axis=1)
y = cancer.diagnosis
vif_df = VIF(X).sort_values('VIF', ascending = False, ignore_index = True)
print(vif_df.head(8))

# Removing features with VIF > 10,000

high_vif_features = list(vif_df.Predictor.iloc[:2])
vif_features = X.drop(high_vif_features, axis=1)

X_train, X_test, y_train, y_test = train_test_split(vif_features, y, test_size = 0.2, random_state = 39)
#Logistic Regression
steps = [('scaler', StandardScaler()),
         ('log_reg', LogisticRegression())]
pipeline = Pipeline(steps)

parameters = dict(log_reg__solver = ['newton-cg', 'lbfgs', 'liblinear'],
                  log_reg__penalty =  ['l2'],
                  log_reg__C = [100, 10, 1.0, 0.1, 0.01])


cv = GridSearchCV(pipeline,
                  param_grid = parameters,
                  cv = 5,
                  scoring = 'accuracy',
                  n_jobs = -1,
                  error_score = 0.0)

cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
log_accuracy = accuracy_score(y_pred, y_test) * 100

print('\033[1m' +'Best parameters : '+ '\033[0m', cv.best_params_)
print('\033[1m' +'Accuracy : {:.2f}%'.format(log_accuracy) + '\033[0m')
print('\033[1m' +'Classification report : '+ '\033[0m\n', classification_report(y_test, y_pred))

cm = confusion_matrix(y_pred, y_test)
print('\033[1m' +'Confusion Matrix : '+ '\033[0m')
sns.heatmap(cm, cmap = 'OrRd',annot = True, fmt='d')
plt.show()

# KNN with VIF features and hyperparameter tuning

steps = [('scaler', StandardScaler()),
         ('knn', BaggingClassifier(KNeighborsClassifier()))]
pipeline = Pipeline(steps)

parameters = dict(knn__base_estimator__metric = ['euclidean', 'manhattan', 'minkowski'],
                  knn__base_estimator__weights =  ['uniform', 'distance'],
                  knn__base_estimator__n_neighbors = range(2,15),
                  knn__bootstrap = [True, False],
                  knn__bootstrap_features = [True, False],
                  knn__n_estimators = [5])


cv = GridSearchCV(pipeline,
                  param_grid = parameters,
                  cv = 5,
                  scoring = 'accuracy',
                  n_jobs = -1,
                  )

cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
knn_accuracy = accuracy_score(y_pred, y_test) * 100

print('\033[1m' +'Best parameters : '+ '\033[0m', cv.best_params_)
print('\033[1m' +'Accuracy : {:.2f}%'.format(knn_accuracy) + '\033[0m')
print('\033[1m' +'Classification report : '+ '\033[0m\n', classification_report(y_test, y_pred))

cm = confusion_matrix(y_pred, y_test)
print('\033[1m' +'Confusion Matrix : '+ '\033[0m')
sns.heatmap(cm, cmap = 'OrRd',annot = True, fmt='d')
plt.show()

# SVC with VIF features and hyperparameter tuning

steps = [('scaler', StandardScaler()),
         ('svc', SVC())]
pipeline = Pipeline(steps)

parameters = dict(svc__kernel = ['poly', 'rbf', 'sigmoid'],
                  svc__gamma =  [0.0001, 0.001, 0.01, 0.1],
                  svc__C = [0.01, 0.05, 0.5, 0.1, 1, 10, 15, 20])


cv = GridSearchCV(pipeline,
                  param_grid = parameters,
                  cv = 5,
                  scoring = 'accuracy',
                  n_jobs = -1)

cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
svc_accuracy = accuracy_score(y_pred, y_test) * 100

print('\033[1m' +'Best parameters : '+ '\033[0m', cv.best_params_)
print('\033[1m' +'Accuracy : {:.2f}%'.format(svc_accuracy) + '\033[0m')
print('\033[1m' +'Classification report : '+ '\033[0m\n', classification_report(y_test, y_pred))

cm = confusion_matrix(y_pred, y_test)
print('\033[1m' +'Confusion Matrix : '+ '\033[0m')
sns.heatmap(cm, cmap = 'OrRd',annot = True, fmt='d')
plt.show()

# Random Forest Classifier with VIF features and hyperparameter tuning

steps = [('scaler', StandardScaler()),
         ('rf', RandomForestClassifier(random_state = 0))]
pipeline = Pipeline(steps)

parameters = dict(rf__n_estimators = [10,100],
                  rf__max_features = ['sqrt', 'log2'],
)


cv = GridSearchCV(pipeline,
                  param_grid = parameters,
                  cv = 5,
                  scoring = 'accuracy',
                  n_jobs = -1)

cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
rf_accuracy = accuracy_score(y_pred, y_test) * 100

print('\033[1m' +'Best parameters : '+ '\033[0m', cv.best_params_)
print('\033[1m' +'Accuracy : {:.2f}%'.format(rf_accuracy) + '\033[0m')
print('\033[1m' +'Classification report : '+ '\033[0m\n', classification_report(y_test, y_pred))

cm = confusion_matrix(y_pred, y_test)
print('\033[1m' +'Confusion Matrix : '+ '\033[0m')
sns.heatmap(cm, cmap = 'OrRd',annot = True, fmt='d')
plt.show()


# Ridge Classifier with VIF features and hyperparameter tuning

steps = [('scaler', StandardScaler()),
         ('ridge', RidgeClassifier())]
pipeline = Pipeline(steps)

parameters = dict(ridge__alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


cv = GridSearchCV(pipeline,
                  param_grid = parameters,
                  cv = 5,
                  scoring = 'accuracy',
                  n_jobs = -1,
                  error_score = 0.0)

cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
ridge_accuracy = accuracy_score(y_pred, y_test) * 100

print('\033[1m' +'Best parameters : '+ '\033[0m', cv.best_params_)
print('\033[1m' +'Accuracy : {:.2f}%'.format(ridge_accuracy) + '\033[0m')
print('\033[1m' +'Classification report : '+ '\033[0m\n', classification_report(y_test, y_pred))

cm = confusion_matrix(y_pred, y_test)
print('\033[1m' +'Confusion Matrix : '+ '\033[0m')
sns.heatmap(cm, cmap = 'OrRd',annot = True, fmt='d')
plt.show()


# Gradient Boosting Classifier  with VIF features  and hyperparameter tuning

steps = [('scaler', StandardScaler()),
         ('gbc', GradientBoostingClassifier())]
pipeline = Pipeline(steps)

parameters = dict(gbc__n_estimators = [10,100,200],
                  gbc__loss = ['deviance', 'exponential'],
                  gbc__learning_rate = [0.001, 0.1, 1, 10]
)


cv = GridSearchCV(pipeline,
                  param_grid = parameters,
                  cv = 5,
                  scoring = 'accuracy',
                  n_jobs = -1,
                  error_score = 0.0
                  )

cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
gb_accuracy = accuracy_score(y_pred, y_test) * 100

print('\033[1m' +'Best parameters : '+ '\033[0m', cv.best_params_)
print('\033[1m' +'Accuracy : {:.2f}%'.format(gb_accuracy) + '\033[0m')
print('\033[1m' +'Classification report : '+ '\033[0m\n', classification_report(y_test, y_pred))

cm = confusion_matrix(y_pred, y_test)
print('\033[1m' +'Confusion Matrix : '+ '\033[0m')
sns.heatmap(cm, cmap = 'OrRd',annot = True, fmt='d')
plt.show()


# Extreme gradient Boosting classifier with VIF features

xgb = XGBClassifier(max_depth = 5,
                        min_child_weight = 1,
                        gamma = 0.3,
                        subsample = 0.8,
                        colsample_bytree = 0.8,
                        learning_rate = 0.1,
                        reg_alpha=0.05,
                        disable_default_eval_metric = True)

xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

xgb_accuracy = accuracy_score(y_pred, y_test) * 100

print('\033[1m' +'Best parameters : '+ '\033[0m', cv.best_params_)
print('\033[1m' +'Accuracy : {:.2f}%'.format(xgb_accuracy) + '\033[0m')
print('\033[1m' +'Classification report : '+ '\033[0m\n', classification_report(y_test, y_pred))

cm = confusion_matrix(y_pred, y_test)
print('\033[1m' +'Confusion Matrix : '+ '\033[0m')
sns.heatmap(cm, cmap = 'OrRd',annot = True, fmt='d')
plt.show()

# Accuracies of models

results = {'Model' :['Logistic Regression', 'KNN', 'SVC', 'Random Forest', 'Rigde Classifier', 'Gradient Boosting', 'XGBoost'],
           'Accuracy' : [log_accuracy, knn_accuracy, svc_accuracy, rf_accuracy, ridge_accuracy, gb_accuracy, xgb_accuracy]}

results = pd.DataFrame(results).sort_values('Accuracy', ignore_index=True, ascending=False)
results.Accuracy = results.Accuracy.round(2)
results

fig = px.line(results,
            x = results.Model,
            y = results.Accuracy,
            text=results.Accuracy,
        )
fig.update_traces(textposition = 'top right')
fig.update_layout(title = 'Model vs Accuracy',
                  plot_bgcolor = '#f9faed')

fig.show()
