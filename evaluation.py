import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import tensorflow as tf

import utils

np.set_printoptions(precision=2)

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

data = np.loadtxt('./data_w3_ex1.csv',delimiter=',')

x = data[:,0]
y = data[:,1]

x = np.expand_dims(x,axis=1)
y = np.expand_dims(y,axis=1)

x_train,x_,y_train,y_ = train_test_split(x,y,test_size=0.40,random_state=1)

x_cv,x_test,y_cv,y_test = train_test_split(x_,y_,test_size=0.50,random_state=1)

del x_,y_

scale_linear = StandardScaler()

X_train_scaled = scale_linear.fit_transform(x_train)

linear_model = LinearRegression()
linear_model.fit(X_train_scaled,y_train)

yhat = linear_model.predict(X_train_scaled)

# print(f"training MSE (using sklearn function): {mean_squared_error(y_train, yhat) / 2}")

total_squared_error = 0

for i in range(len(yhat)):
  squared_error_i = (yhat[i]-y_train[i])**2
  total_squared_error += squared_error_i

mse = total_squared_error/(2*len(yhat))

# print(f"training MSE (for-loop implementation): {mse.squeeze()}")

X_cv_scaled = scale_linear.transform(x_cv)

# print(f"Mean used to scale the CV set: {scaler_linear.mean_.squeeze():.2f}")
# print(f"Standard deviation used to scale the CV set: {scale_linear.scale_.squeeze():.2f}")

yhat = linear_model.predict(X_cv_scaled)

# print(f"Cross validation MSE: {mean_squared_error(y_cv, yhat) / 2}")


poly = PolynomialFeatures(degree=2,include_bias=False)

X_train_mapped = poly.fit_transform(x_train)

# print(X_train_mapped[:5])

scaler_poly = StandardScaler()

X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)

# print(X_train_mapped_scaled[:5])

model = LinearRegression()

model.fit(X_train_mapped_scaled,y_train)

yhat = model.predict(X_train_mapped_scaled)

X_cv_mapped = poly.transform(x_cv)

X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

yhat = model.predict(X_cv_mapped_scaled)

# print(f"Cross validation MSE: {mean_squared_error(y_cv, yhat) / 2}")

train_mses = []
cv_mses = []
models=[]
polys = []
scalers = []

for degree in range(1,11):

  #add poly features
  poly = PolynomialFeatures(degree,include_bias=False)
  X_train_mapped = poly.fit_transform(x_train)
  polys.append(poly)

  #Scale training set
  scaler_poly = StandardScaler()
  X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
  scalers.append(scaler_poly)

  #Create and train model
  model = LinearRegression()
  model.fit(X_train_mapped_scaled, y_train)
  models.append(model)

  #Compute training MSE
  yhat = model. predict(X_train_mapped_scaled)
  train_mse = mean_squared_error(y_train,yhat)/2
  train_mses.append(train_mse)

  #Add polynomial features and scale CV set
  X_cv_mapped = poly.transform(x_cv)
  X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

  #Compute CV MSE
  yhat = model.predict(X_cv_mapped_scaled)
  cv_mse = mean_squared_error(y_cv,yhat)/2
  cv_mses.append(cv_mse)

degrees= range(1,11)
# utils.plot_train_cv_mses(degrees, train_mses,cv_mses,title ="degree of polynomial vs train and CV MSEs")

# degree = np.argmin(cv_mses) + 1
# print(f"lowest CV MSE degree is {degree}")


X_test_mapped = polys[degree-1].transform(x_test)

X_test_mapped_scaled = scalers[degree-1].transform(X_test_mapped)

yhat = models[degree-1].predict(X_test_mapped_scaled)
test_mse = mean_squared_error(y_test,yhat)/2

# print(f"Training MSE: {train_mses[degree-1]:.2f}")
# print(f"Cross Validation MSE: {cv_mses[degree-1]:.2f}")
# print(f"Test MSE: {test_mse:.2f}")


#Preparing the Data

degree = 1

poly = PolynomialFeatures(degree,include_bias=False)
X_train_mapped = poly.fit_transform(x_train)
X_cv_mapped = poly.transform(x_cv)
X_test_mapped = poly.transform(x_test)


scaler = StandardScaler()

X_train_mapped_scaled = scaler.fit_transform(X_train_mapped)
X_cv_mapped_scaled = scaler.transform(X_cv_mapped)
X_test_mapped_scaled = scaler.transform(X_test_mapped)

nn_train_mses = []
nn_cv_mses = []

nn_models = utils.build_models()

for model in nn_models:

  model.compile(
    loss='mse',
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1),
  )

  print(f"Training {model.name}...")

  model.fit(
    X_train_mapped_scaled,y_train,
    epochs = 300,
    verbose = 0
  )

  print("Done!\n")


  #Record training MSEs 

  yhat = model.predict(X_train_mapped_scaled)
  train_mse = mean_squared_error(y_train,yhat)/2
  nn_train_mses.append(train_mse)

  #Record the cross validation MSEs

  yhat = model.predict(X_cv_mapped_scaled)
  cv_mse = mean_squared_error(y_cv,yhat)/2
  nn_cv_mses.append(cv_mse)

# print("RESULTS: ")
# for model_num in range(len(nn_train_mses)):
#   print(
#     f"Model {model_num+1}: Training MSE: {nn_train_mses[model_num]:.2f}, " +
#         f"CV MSE: {nn_cv_mses[model_num]:.2f}"
#         )
  

model_num = 3

yhat = nn_models[model_num-1].predict(X_test_mapped_scaled)
test_mse = mean_squared_error(y_test,yhat)/2

# print(f"Selected Model: {model_num}")
# print(f"Training MSE: {nn_train_mses[model_num-1]:.2f}")
# print(f"Cross Validation MSE: {nn_cv_mses[model_num-1]:.2f}")
# print(f"Test MSE: {test_mse:.2f}")



# -----MODEL EVALUATION FOR CLASSIFICATION TASKS---- #

data = np.loadtxt('./data_w3_ex2.csv',delimiter=',')

x_bc = data[:,:-1]
y_bc = data[:,-1]

y_bc = np.expand_dims(y_bc,axis=1)

utils.plot_bc_dataset(x=x_bc,y=y_bc,title='x1 vs .x2')


from sklearn.model_selection import train_test_split

# Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables.
x_bc_train, x_, y_bc_train, y_ = train_test_split(x_bc, y_bc, test_size=0.40, random_state=1)

# Split the 40% subset above into two: one half for cross validation and the other for the test set
x_bc_cv, x_bc_test, y_bc_cv, y_bc_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

# Delete temporary variables
del x_, y_

scaler_linear = StandardScaler()

# Compute the mean and standard deviation of the training set then transform it
x_bc_train_scaled = scaler_linear.fit_transform(x_bc_train)
x_bc_cv_scaled = scaler_linear.transform(x_bc_cv)
x_bc_test_scaled = scaler_linear.transform(x_bc_test)

probabilities = np.array([0.2,0.6,0.7,0.3,0.8])

#Apply threshold.If greater than 0.5 set to 1, else 0
predictions = np.where(probabilities>=0.5,1,0)

ground_truth = np.array([1,1,1,1,1])

misclassifed = 0

num_predictions = len(predictions)

for i in range(num_predictions):
  if predictions[i]!= ground_truth[i]:
    misclassifed+=1

fraction_error = misclassifed/num_predictions

print(f"probabilities: {probabilities}")
print(f"predictions with threshold=0.5: {predictions}")
print(f"targets: {ground_truth}")
print(f"fraction of misclassified data (for-loop): {fraction_error}")
print(f"fraction of misclassified data (with np.mean()): {np.mean(predictions != ground_truth)}")


nn_train_error = []
nn_cv_error = []

models_bc = utils.build_models()

for model in models_bc:

  model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logist=True),
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01),
  )

  print(f"Training {model.name}...")

  model.fit(
    x_bc_train_scaled,y_bc_train,
    epochs = 200,
    verbose=0

  )

  print("Done!\n")

  threshold = 0.5

  #fraction of misclassified exmaples for training set
  yhat = model.predict(x_bc_test_scaled)
  yhat = tf.math.sigmoid(yhat)
  yhat = np.where(yhat >= threshold,1,0)
  train_error = np.mean(yhat != y_bc_train)
  nn_train_error.append(train_error)

  # Record the fraction of misclassified examples for the cross validation set
  yhat = model.predict(x_bc_cv_scaled)
  yhat = tf.math.sigmoid(yhat)
  yhat = np.where(yhat >= threshold, 1, 0)
  cv_error = np.mean(yhat != y_bc_cv)
  nn_cv_error.append(cv_error)

# Print the result
for model_num in range(len(nn_train_error)):
    print(
        f"Model {model_num+1}: Training Set Classification Error: {nn_train_error[model_num]:.5f}, " +
        f"CV Set Classification Error: {nn_cv_error[model_num]:.5f}"
        )
    
# Select the model with the lowest error
model_num = 3

# Compute the test error
yhat = models_bc[model_num-1].predict(x_bc_test_scaled)
yhat = tf.math.sigmoid(yhat)
yhat = np.where(yhat >= threshold, 1, 0)
nn_test_error = np.mean(yhat != y_bc_test)

print(f"Selected Model: {model_num}")
print(f"Training Set Classification Error: {nn_train_error[model_num-1]:.4f}")
print(f"CV Set Classification Error: {nn_cv_error[model_num-1]:.4f}")
print(f"Test Set Classification Error: {nn_test_error:.4f}")

