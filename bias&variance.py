from sklearn.linear_model import LinearRegression,Ridge

import utils

x_train,y_train,x_cv,y_cv,x_test,y_test = utils.prepare_dataset('./c2w3_lab2_data1.csv')

model = LinearRegression()

utils.train_plot_poly(model, x_train, y_train, x_cv, y_cv, max_degree=10, baseline=400)

utils.train_plot_poly(model, x_train, y_train, x_cv, y_cv, max_degree=10, baseline=250)

x_train, y_train, x_cv, y_cv, x_test, y_test = utils.prepare_dataset('data/c2w3_lab2_data2.csv')

model = LinearRegression()

utils.train_plot_poly(model, x_train, y_train, x_cv, y_cv, max_degree=6, baseline=250)

reg_params = [10,5,2,1,0.5,0.2,0.1]

utils.train_plot_reg_params(reg_params, x_train, y_train, x_cv, y_cv, degree= 4, baseline=250)

# You can change the same code to deal with high variance.In that case you have to decrease lamda