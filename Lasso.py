#Introduction to two new methods of Regression: Ridge Regression and Lasso Regression and the way they work
#The dataset was taken from www.gapminder.org. The dataset contains various features like country's population,fertility,HIV rate
#,co2 emmission,BMI of male and female child mortality here i'm predicting life expetancy of one using different this different
#features since it is quantitative i am fitting regression model to this dataset. and introducing two new topics which were not
#covered in class

#Importing All Necessary Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

#Apart from doing Linear regression i have fit my data to two other type of regularized regression and explained them
# 1. Lassso Regression 2. Ridge Regression
#lasso performs regularization by adding to the loss function a penalty term of the absolute value of each coefficient
#multiplied by some alpha(Hyperparamter).This is also known as L1 regularization because the regularization term is the L1

#Ridge Regression : it does sum of the squared values of the coefficients multiplied by some alpha(Hyperparameter). This is called
#L2 Regularizartion


#Reading the data from csv file
data = pd.read_csv("D:/578D/GradeBonus/gm_2008_region.csv")


#Here, dropping the region column as it is non numeric and doesnt have that much impact on the result.
data = data.drop('Region', axis=1)
data.head()

#it is important to explore the data before building a model, checking the correlation of all features through heatmap
#cells in green shows positive correlation and red are in negative correlation from map we can see that GDp,BMI male are in positive correlation
# whereas, life and fertility are negatively correlated
sns.heatmap(data.corr(),square=True,cmap='RdYlGn')
plt.show()

#Here I have checked that fertility and life are in strong negative correlation let's fit linear regression model to it
#extraction the data as x and y from the dataframe
x = data['fertility']
y = data['life']

print(x.shape)
print(y.shape)

x = x.values.reshape(-1,1)
y = y.values.reshape(-1,1)

print(x.shape)
print(y.shape)

#Plotting the scatter plot of various points of x and y
plt.scatter(x,y)

#generating some random data after fitting the model to predict the data and calculate the Regression score.
#generating the regression line to show that how model has learnt from the dataset provided to it.
reg = LinearRegression()
#generating the data for prediction
pred = np.linspace(min(x), max(x)).reshape(-1,1)
reg.fit(x,y)
y_pred = reg.predict(pred)
print(reg.score(x, y))
# Plot regression line
plt.plot(pred, y_pred, color='black', linewidth=4)
plt.show()



#It is always better to build a model from the full dataset. hence, i am trying to fit in full data set.
#In addition to computing the R2 score of regression model i'm also computing the root mean squared error for this model.
#extracting the label from the full data
X_full = data.drop(['life'], axis=1)
y_full = data['life'].values.reshape(-1,1)
print(X_full.shape)
print(y_full.shape)
#extracting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size = 0.3, random_state=42)
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print("R2: {}".format(reg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error: {}".format(rmse))
#Here if we see R2 is dependent on the train test split and we can not generalize the model performance by this method
# so we will do cross validation on this data so that all data is covered and we  will get proper accuracy of the model
#if we are taking 10 folds then we will get 5 differnt values of R2 by this way we can compute the average R2 score of the model.
#It maximizes the amount of data that is used to train the model

#doing 10 fold cross validation on the data
cv_scores = cross_val_score(reg,X_full,y_full,cv=10)
print(cv_scores)
print("Average 10-Fold CV Score: {}".format(np.mean(cv_scores)))

#Linear Regression Minimizes the loss function. It choses co-efficient for each feature variables.
#and Large coeffients can lead to overfitting of the model to overcome this Regularaized regression is used

#Regularized Regression: Regularizing is the method of penalizing the large coeffients

#here i will introduce two regularized regression methods and expalain it.
#Lasso Regressor : can be used to select important features of the dataset.
#It shrinks the coefficients of the less important features to exactly 0

data_columns = X_full.columns.tolist()
lasso = Lasso(alpha=0.4,normalize=True)
lasso.fit(X_full,y_full)
#Printing the coefficients of the equation
lasso_coef = lasso.coef_
print(lasso_coef)
#Plotting the coefficients
#as we have 8 columns we will set length as 8
length = 8
plt.plot(range(length), lasso_coef)
plt.xticks(range(length), data_columns, rotation=60)
plt.margins(0.02)
plt.show()
#here we can see from the plot and data only co-efficient for child mortality is set to some value
#and all other co-efficients are set to zero


#In this function doing Ridge Regression with different alphas so that i can choose the best alpha for this model
#plotting the R2 score as well as standard error for each alpha

#getting different values of cvscore and standard deviation
def display_plot(cv_scores, cv_scores_std):
    #plotting the new figure
    fig = plt.figure()
    #adding a subplot so that we can add figure
    ax = fig.add_subplot(1,1,1)
    #plotting alpha space vs cv scores
    ax.plot(alpha_space, cv_scores)
    #setting up standard error
    std_error = cv_scores_std / np.sqrt(10)
    #filling the cv_scores with standard error with little bit transperancy so that we can see the error
    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    #setting one horizontal line at maximum cv score so that we can get best value of alpha at intersection
    ax.axhline(np.max(cv_scores), linestyle='--')
    #setting x axis limit to full alpha space array
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    #Setting x axis scale to log
    ax.set_xscale('log')
    plt.show()


# R2 Score with Different Alpha to see how the R2 score varies with different alphas
# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Creating the ridge regressor
ridge = Ridge(normalize=True)

# Computing scores over range of alpha
for alpha in alpha_space:
    # current alpha value
    ridge.alpha = alpha

    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X_full, y_full, cv=10)

    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))

    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)
print(ridge_scores)
print(ridge_scores_std)


