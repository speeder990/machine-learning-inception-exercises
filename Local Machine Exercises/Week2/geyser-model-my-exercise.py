import csv
import time
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

# Note: Original comments from the example were retained.

# ################################
# #####  Our Custom Methods ######
# ################################

# Method to estimate the coefficients
# Detailed Derivation of beta_1 and beta_0 estimation is present at the url:
# https://are.berkeley.edu/courses/EEP118/current/derive_ols.pdf

def slr(X, Y):
    intermediate_beta = []
    beta_1_numerator_sum = 0.0
    beta_1_denominator_sum = 0.0

    for i in range(0, len(X)):
        beta_1_numerator_sum = beta_1_numerator_sum + ((X[i] - np.mean(X))*(Y[i]- np.mean(Y)))
        beta_1_denominator_sum = beta_1_denominator_sum + ((X[i] - np.mean(X))**2)

        beta_1 = beta_1_numerator_sum/beta_1_denominator_sum
        beta_0 = np.mean(Y) - ((beta_1)*np.mean(X))

    return [beta_0, beta_1, intermediate_beta]

# Method to predict response variable Y (in this case interval before the next erruption) for new values
# of X (in this case duration of eruption) using the estimated coefficients.


def predict(coef, X):
    beta_0 = coef[0]
    beta_1 = coef[1]

    # Our Regression Model defined using the coefficients from slr function
    Y = beta_0 + (beta_1 * X)

    return Y

# ####################################
# ##### Reading Data from dataset ####
# ####################################


# Opens the file handler for the dataset file. Using variable 'f' we can access and manipulate our file
# anywhere in our code
f = open('Dataset/faithful.csv', 'r+')

# Predictors Collection (or your input variable) (which in this case is just the duration of eruption)
X = []

# Output Response (or your output variable) (which in this case is the duration after which next eruption
# will occur)
Y = []

# Initializing a reader generator using reader method from csv module. A reader generator takes each line
# from the file
# and converts it into list of columns.
reader = csv.reader(f)

# Using for loop, we are able to read one row at a time
for row in reader:
    if row[1] != "Duration":
        X.append(float(row[1]))
        Y.append(float(row[2]))

# Close the file once we have successfully stored all data into our X and Y variables
f.close()

# Visualize the data using Scatter plot of matplotlib library. A scatter plot is a plot between two continuous variables.
# and it helps us in determining the relationship between those two continuous variables.
# For more information on working of scatter plot function of matplotlib - you can visit the following url:
# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html

# ###########################
# ### Data Visualization ####
# ###########################

# Visualize the data using Scatter plot of matplotlib library. A scatter plot is a plot between two continuous variables
# and it helps us in determining the relationship between those two continuous variables.


plt.scatter(X, Y, s=2)
plt.xlabel("Duration of Eruption (in minutes)")
plt.ylabel("Time duration before the next eruption (in minutes)")
plt.show()

# ####################################################
# ### Model Training (or coefficient estimation) #####
# ####################################################
# Using our slr function we estimate coefficients of our regression line. The slr function returns a list
# of coefficients

coefficients = slr(X,Y)

# #########################
# ## Making Predictions ###
###########################

# Using our predict function and the coefficients given by our slr function we can now predict the time it will
# take for the next eruption.
last_eruption_duration = float(input("Duration of the last eruption (in minutes):"))
print("Time it will take for the next eruption to occur (in minutes): ", predict(coefficients,last_eruption_duration))

# ###########################
# #### Error Calculation ####
# ###########################

print("\n\nAccuracy Metrics of the model\n----------------------------------------------")

# Calculation of RSE
RSS = 0
for idx in range(0, len(X)):
    actual_y = Y[idx]
    predicted_y = predict(coefficients, X[idx])
    RSS = RSS + ((actual_y - predicted_y)**2)
RSE = np.sqrt((1/float(len(X)-2))*RSS)

print("Residual Standard Error:", RSE)
print("% Residual Standard Error (over average interval):", (RSE/np.mean(Y))*100)

# Calculation of R_Squared
TSS = 0
for idx in range(0, len(X)):
    actual_y = Y[idx]
    TSS = TSS + ((actual_y - np.mean(Y))**2)
R_Squared = (TSS - RSS)/ TSS

print("\nR-Squared Value:", R_Squared)
