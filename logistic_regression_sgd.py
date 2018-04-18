#!/usr/bin/env python

# Run logistic regression training for different learning rates with stochastic gradient descent.
import scipy.special as sps
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import collections
import operator

# Step size for gradient descent.
etas = [0.5, 0.3, 0.1, 0.05, 0.01]
#etas = [0.1, 0.05, 0.01]


#execfile("logistic_regression_mod.py")

def SGD(X, t,max_iter = 500,tol = 0.00001):
    X = X.as_matrix()
    t = t.as_matrix()

    all_errors = dict()
    # Randomly permute the training data.
    #data = np.random.permutation(all_data)
    n_train = np.shape(t)[0]
    final_weight_epoch= dict()
    min_error = dict()

    for eta in etas:
        # Initialize w.
        w_vector = np.zeros(train_data.shape[1])
        w_vector[0]=0.1
        #print w_vector.shape
        e_all = []
        iterations = []

        for iter in range (0, max_iter):
            # print('Epoch {}'.format(iter))
            for n in range (0, n_train):
                # Compute output using current w on sample x_n.
                y = sps.expit(np.dot(X[n,:],w_vector))

                # Gradient of the error, using Assignment result
                #print y

                grad_e = (y - t[n])*(X[n,:]) / n_train

                # Update w, *subtracting* a step in the error derivative since we're minimizing
                w_vector = w_vector - eta * grad_e

            # Compute error over all examples, add this error to the end of error vector.
            # Compute output using current w on all data X.
            y = sps.expit(np.dot(X,w_vector))

            # e is the error, negative log-likelihood (Eqn 4.90)
            e = 0
            for index, i in enumerate(t):
                if i ==0:
                    e += np.log(1-y[index])
                else:
                    e += np.log(y[index])
            # e = -np.mean(np.multiply(t,np.log(y)) + np.multiply((1-t),np.log(1-y)))
            e_all.append(-e/n_train)
            iterations.append(iter)

            # Print some information.
            # print 'eta={0}, epoch {1:d}, negative log-likelihood {2:.4f}, w={3}'.format(eta, iter, e, w_vector.T)

            # Stop iterating if error doesn't change more than tol.
            if iter>0:
                if np.absolute(e-e_all[iter-1]) < tol:
                    break

        min_error[eta]=-e/n_train
        best_eta= min(min_error.items(),key=operator.itemgetter(1))[0]

        all_errors[eta]= {'error': e_all,'iter':iterations}
        final_weight_epoch[eta]=w_vector
    return all_errors,final_weight_epoch,best_eta


data = pd.read_csv("data.csv")
columns_to_be_removed = ['id', 'PlayerName', 'sum_7yr_TOI', 'DraftYear', 'Country', 'GP_greater_than_0', 'Overall',
                         'sum_7yr_GP']
discrete_columns = ['country_group', 'Position']
country_group_values = data['country_group'].unique()
position_values = data['Position'].unique()
all_discrete_values = list(country_group_values) + list(position_values)
test_data = data[data['DraftYear'] == 2007]
train_data = data[(data['DraftYear'] >= 2004) & (data['DraftYear'] <= 2006)]
y_column = 'GP_greater_than_0'

class_output_train = pd.DataFrame(train_data[y_column])
class_output_test = pd.DataFrame(test_data[y_column])



for col in columns_to_be_removed:
    del train_data[col]
    del test_data[col]

for col in discrete_columns:
    dummy_col = pd.get_dummies(train_data[col])
    train_data = pd.concat([train_data, dummy_col], axis=1)

    dummy_col_test = pd.get_dummies(test_data[col])
    test_data = pd.concat([test_data, dummy_col_test], axis=1)
    del train_data[col]
    del test_data[col]


def standardize_predictors(col):
    mean = np.mean(col)
    std = np.std(col)
    cols = col.name.split('_by_')
    if col.name not in all_discrete_values:
        if (cols[0] not in all_discrete_values) or (cols[1] not in all_discrete_values):
            col = (col - mean) / std
    return col

def boolean_converter(row):
    if row.values[0] == 'yes':
        return 1
    else:
        return 0


class_output_train= class_output_train.apply(boolean_converter,axis=1)
class_output_test= class_output_test.apply(boolean_converter, axis=1)

train_data = train_data.apply(standardize_predictors, axis=0)
train_data.insert(0,"bais_weight",1)
test_data = test_data.apply(standardize_predictors, axis=0)
test_data.insert(0, "bais_weight_test", 1)

# print train_data.shape
# print test_data.shape
# print class_output_train.shape
# print class_output_test.shape


# Data matrix, with column of ones at end.
X = train_data
# Target values, 0 for class 1, 1 for class 2.
t = class_output_train
# # For plotting data
# class1 = np.where(t==0)
# X1 = X[class1]
# class2 = np.where(t==1)
# X2 = X[class2]

# Error values over all iterations.

all_errors, final_weight_epoch, best_eta = SGD(train_data,class_output_train)

# # Plot error over iterations for all etas
plt.figure(10)
plt.rcParams.update({'font.size': 15})
for eta in sorted(all_errors):
    plt.plot( all_errors[eta]['iter'], all_errors[eta]['error'],label='sgd eta={}'.format(eta))

plt.ylabel('Negative log likelihood')
plt.title('Training logistic regression with SGD')
plt.xlabel('Epoch')
#plt.axis([0, 500, 0.2, 0.7])
plt.legend()
plt.show()

def find_accuracy(w_vector, test_data, class_output_test):
    test_data = test_data.as_matrix()
    class_output_test= class_output_test.as_matrix()
    test_count = class_output_test.size
    predictions=0.0
    for iter in range(0, test_count):
        output = sps.expi(np.dot(test_data[iter,:], w_vector))
       # print output, class_output_test[iter]
        if output >=0.5 and class_output_test[iter]==1:
            predictions +=1
        if output < 0.5 and class_output_test[iter]==0:
            predictions +=1

    accuracy = predictions/test_count
    return accuracy

w_vector_accuracy = final_weight_epoch[best_eta]
acc =find_accuracy(w_vector_accuracy,test_data,class_output_test)
print "Accuracy is  %.6f percent "% (acc*100)
