import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

from utils import Model,plot_VDD,plot_W_n,plot_C_L

x=pd.read_csv('../data/Input_data_Full_Adder.csv',index_col=0)
y=pd.read_csv('../data/Output_data_Full_Adder.csv',index_col=0)

model=Model()
early_stopping = tf.keras.callbacks.EarlyStopping( monitor='val_loss', patience=15, restore_best_weights=True)

model.train(np.array(x),np.array(y),callbacks=[early_stopping])

model.evaluate()

x_test,y_test=model.get_test_data()
y_test

y_test_prediction=model.predict(x_test)
y_test_prediction

relative_errors = 100*np.abs((y_test - y_test_prediction) / y_test)

plt.scatter(x_test[:,1], relative_errors[:,1])
plt.xlabel("VDD/W_n/C_L")
plt.ylabel("Relative Error")
plt.title("Relative Error Distribution")
plt.show()

bounds = [(1.5, 2.5),  # VDD bounds
          (0.2e-6, 5e-6),  # W_n bounds
          (1e-13, 1e-12)]  # C_L bounds

# Define the objective function
def objective_function(params):
    
    VDD, W_n, C_L = params
    prediction = model.predict([[VDD,W_n,C_L]])
    power, delay = prediction[0]
    
    return delay

# Perform optimization
result = differential_evolution(objective_function,bounds=bounds,strategy='best1bin', maxiter=250, popsize=5 )

# Extract optimal values
optimal_VDD, optimal_W_n, optimal_C_L = result.x
print(f"Optimal VDD: {optimal_VDD}, Optimal W_n: {optimal_W_n}, Optimal C_L: {optimal_C_L}")

num_samples = 1000
VDD_range = [1.5, 2.5]        # VDD: 1.5V to 2.5V
Wn_range = [0.18e-6, 5e-6]     # W/L: 0.18µm to 5µm
CL_range = [2e-14, 1e-12]     # C_L: 20fF to 1pF

VDD=1.8
W_n=1e-6
C_L=0.5e-12

plot_VDD(model,VDD_range=VDD_range,W_n=W_n,C_L=C_L)
plot_W_n(model,Wn_range=Wn_range,VDD=VDD,C_L=C_L)
plot_C_L(model,CL_range=CL_range,VDD=VDD,W_n=W_n)