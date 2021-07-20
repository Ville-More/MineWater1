import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


#Load data

rf_model = pd.read_csv('ML_Model_Final.csv')

print(rf_model)

#Identify dependent (y) and independent variables (X).
#Dependent variable will be EC and pH while the independent variables will be Turbidity, TDS, SO4 and Fe

X = rf_model[['Turbidity', 'TotalDissolvedSolids', 'Sulfate', 'Iron']]
y = rf_model[['Electrical_Conductivity', 'pH']]

print(X)
print(y)

#Splitting the dataset into the Training set and Test set
#Data set will be split using test_size=0.2, meaning 20% of data rows will only be used as test set and the remaining rows will be used as training set for building the Random Forest Regression Model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Training the Random Forest Regression model on the whole dataset
#To train the model, we import the RandomForestRegressor class and assign it to the variable regressor.
#We then use the .fit() function to fit the X_train and y_train values to the regressor.
#We will use 10 trees via n_estimators = 10.

regressorRF = RandomForestRegressor(n_estimators = 10, random_state = 0)

# Fit the model to the training data

regressorRF.fit(X_train, y_train)

#Saving the model to disk

pickle.dump(regressorRF, open('Final_MultiOutputRF_model.pkl', 'wb'))