
# coding: utf-8

# In[11]:


# Import libraries necessary for this project
import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit

# Import supplementary visualizations code visuals.py
import visuals as vs

# Display for notebooks
get_ipython().run_line_magic('matplotlib', 'inline')

# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
    
print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))



# In[12]:


# Minimum price of the data
minimum_price = np.min(prices)

# Maximum price of the data
maximum_price = np.max(prices)

# Mean price of the data
mean_price = np.mean(prices)

# Median price of the data
median_price = np.median(prices)

# Standard deviation of prices of the data
std_price = np.std(prices)

# Show the calculated statistics
print( "Statistics for Boston housing dataset:\n")
print ("Minimum price: ${:,.2f}".format(minimum_price))
print( "Maximum price: ${:,.2f}".format(maximum_price))
print ("Mean price: ${:,.2f}".format(mean_price))
print ("Median price ${:,.2f}".format(median_price))
print ("Standard deviation of prices: ${:,.2f}".format(std_price))


# In[13]:


# Import 'r2_score'
from sklearn.metrics import r2_score
def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    # Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true , y_predict)
    
    # Return the score
    return score


# In[14]:


# Calculate the performance of this model
score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
print("Model has a coefficient of determination, R^2, of {:.3f}.".format(score))


# In[15]:


#vs.ModelComplexity(X_train, y_train)


# Identifying when a model is suffering from high bias or high variance.
# 
# It is easy to identify whether the model is suffering from a high bias or a high variance.
# High variance models have a gap between the training and validation scores.
# This is because it is able to fit the model well but unable to generalize well resulting in a high training score but low validation score.
# High bias models have have a small or no gap between the training and validations scores.
# This is because it is unable to fit the model well and unable to generalize well resulting in both scores converging to a similar low score.
# 
# Maximum depth of 1: High Bias
# 
# Both training and testing scores are low.
# There is barely a gap between the training and testing scores.
# This indicates the model is not fitting the dataset well and not generalizing well hence the model is suffering from high bias.
# 
# Maximum depth of 10: High Variance
# 
# Training score is high. Testing score is low
# There is a substantial gap between the training and testing scores.
# This indicates the model is fitting the dataset well but not generalizing well hence the model is suffering from high variance.

# In[16]:


# Import 'train_test_split'
from sklearn.cross_validation import train_test_split
# Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test =train_test_split(features,prices, test_size = 0.20, random_state=10)
# Success
print("Training and testing split was successful.")


# In[17]:


# Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """

    # Cross-validation sets from the training data
    # X.shape[0] is the total number of elements
    # n_iter is the number of re-shuffling & splitting iterations.
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)

    # Decision tree regressor object
    # Instantiate
    regressor = DecisionTreeRegressor(random_state=0)

    # Dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = dict(max_depth=[1,2,3,4,5,6,7,8,9,10])

    # Transform 'performance_metric' into a scoring function using 'make_scorer' 
    # We initially created performance_metric using R2_score
    scoring_fnc = make_scorer(performance_metric)

    # grid search object
    grid = GridSearchCV(regressor, params, cv=cv_sets, scoring=scoring_fnc) 

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


# In[18]:


# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)
# Produce the value for 'max_depth'
print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))


# In[19]:


# Produce a matrix for client data
client_data = [[5, 17, 6], # Client 1
               [4, 32, 18], # Client 2
               [8, 3, 12]]  # Client 3

# Show predictions
for i, price in enumerate(reg.predict(client_data)):
    print("Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price))

