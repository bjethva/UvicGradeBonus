import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso

data = pd.read_csv("D:/578D/GradeBonus/gm_2008_region.csv")
data = data.drop('Region', axis=1)
data.head()

x = data['fertility']
y = data['life']

print(x.shape)
print(y.shape)

x = x.values.reshape(-1,1)
y = y.values.reshape(-1,1)

print(x.shape)
print(y.shape)

#it is important to explore your data before building models, Let's check data correlation before building a model
sns.heatmap(data.corr(),square=True,cmap='RdYlGn')
plt.show()

X_fertility = data['fertility'].values.reshape(-1,1)
plt.scatter(X_fertility,y)

reg = LinearRegression()
prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)
reg.fit(X_fertility,y)
y_pred = reg.predict(prediction_space)
print(reg.score(X_fertility, y))

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=4)
plt.show()

#In addition to computing the R2 score, we will also compute the Root Mean Squared Error (RMSE)
X_full = data.drop(['life'], axis=1)
y_full = data['life'].values.reshape(-1,1)
print(X_full.shape)
print(y_full.shape)
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size = 0.3, random_state=42)
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print("R2: {}".format(reg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error: {}".format(rmse))
#Here if we see R2 is dependent on the train test split and we can not generalize the model performance by this method
# so we will do cross validation on this data so that all data is covered and we  will get proper accuracy of the model
#if we are taking 10 folds then we will get 5 differnt values of R2 by this way we can compute the average R2 score of the model.It maximizes the amount of data that is used to train the model

cv_scores = cross_val_score(reg,X_full,y_full,cv=10)
print(cv_scores)
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))


#Regularized Regression
#Linear Regression Minimizes the loss function. It choses co-efficient for each feature variables. so, Large coeffients can lead to overfitting of the model
#Regularizing is the method of penalizing the large coeffients
#Lasso Regressor : can be used to select important features of the dataset. It shrinks the coefficients of the less important features to exactly 0
data_columns = X_full.columns.tolist()
lasso = Lasso(alpha=0.4,normalize=True)
lasso.fit(X_full,y_full)
lasso_coef = lasso.coef_
print(lasso_coef)
#Plotting the coefficients
#as we have 8 columns we will set length as 8
length = 8
plt.plot(range(length), lasso_coef)
plt.xticks(range(length), data_columns, rotation=60)
plt.margins(0.02)
plt.show()

#lasso is great for feature selection but when building regression model Ridge regression should be the first choice
#Recall that lasso performs regularization by adding to the loss function a penalty term of the absolute value of each coefficient multiplied by some alpha. This is also known as L1 regularization because the regularization term is the L1

#norm of the coefficients. This is not the only way to regularize, however.

#If instead you took the sum of the squared values of the coefficients multiplied by some alpha - like in Ridge regression - you would be computing the L2
#norm.

#you will practice fitting ridge regression models over a range of different alphas, and plot cross-validated R2 scores for each, using this function that we have defined for you, which plots the R2 score as well as standard error for each alpha:

def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()