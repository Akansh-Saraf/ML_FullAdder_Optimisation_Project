import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


class Model:
    def __init__(self):
        self.model_1=tf.keras.Sequential([
            tf.keras.Input(shape=[3,]),
            #tf.keras.layers.Dense(512,activation='relu'),
            tf.keras.layers.Dense(256,activation='relu'),
            tf.keras.layers.Dense(128,activation='relu'),
            tf.keras.layers.Dense(64,activation='relu'),
            tf.keras.layers.Dense(32,activation='relu'),
            tf.keras.layers.Dense(2,activation='linear')
        ])
        self.model_1.compile(optimizer='adam',loss='mse',metrics=['mae'])
        
    def get_test_data(self):
        x_test_descaled=self.scaler_x.inverse_transform(self.x_test)
        y_test_descaled=self.scaler_y.inverse_transform(self.y_test)
        return x_test_descaled,y_test_descaled
        
    def scale(self,x,y):
        self.x=x
        self.y=y
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        x_scaled = self.scaler_x.fit_transform(x)
        y_scaled = self.scaler_y.fit_transform(y)

        x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)
        self.x_test=x_test
        self.y_test=y_test
        return (x_train, x_test, y_train, y_test)
    
    def train(self,x,y,callbacks):
        x_train, x_test, y_train, y_test=self.scale(x,y)
        self.model_1.fit(x_train,y_train,epochs=500,batch_size=16,verbose=1, validation_split=0.2,callbacks=callbacks)
        
    def predict(self,x_input):
        x_scaled=self.scaler_x.transform(x_input)
        y_scaled=self.model_1.predict(x_scaled)
        y_output=self.scaler_y.inverse_transform(y_scaled)
        return y_output
    
    def evaluate(self):
        test_data=self.scaler_y.inverse_transform(self.y_test)
        test_predictions=self.scaler_y.inverse_transform(self.model_1.predict(self.x_test))

        test_loss, test_mae = self.model_1.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

        relative_error = np.mean(np.abs((test_data - test_predictions) / test_data))

        print("Relative Percentage Error:", relative_error*100)
        
        r2 = r2_score(test_data, test_predictions)
        print("R-squared (multi-output):", r2)
        
        r2_power = r2_score(test_data[:, 0], test_predictions[:, 0]) 
        r2_delay = r2_score(test_data[:, 1], test_predictions[:, 1])  

        print("R-squared for Power:", r2_power)
        print("R-squared for Delay:", r2_delay)


# Define parameter ranges
num_samples = 1000
VDD_range = [1.5, 2.5]        # VDD: 1.5V to 2.5V
Wn_range = [0.18e-6, 5e-6]     # W/L: 0.18µm to 5µm
CL_range = [2e-14, 1e-12]     # C_L: 20fF to 1pF

# Varying VDD
def plot_VDD(model,VDD_range=VDD_range,W_n=1e-6,C_L=.5e-12):
    VDD=np.linspace(*VDD_range,200)
    W_n=np.array(200*[W_n])
    C_L=np.array(200*[C_L])
    inputs=pd.DataFrame({'VDD':VDD,'W_n':W_n,'C_L':C_L})
    predictions=model.predict(np.array(inputs))
    delay,power=predictions[:,0],predictions[:,1]

    # Scatter plot for power
    plt.subplot(3,2,1)
    plt.plot(VDD, power)
    plt.xlabel('VDD (V)')
    plt.ylabel('Power (W)')
    plt.title('Power vs VDD')
    
    # Scatter plot for delay
    plt.subplot(3,2,2)
    plt.plot(VDD, delay)
    plt.xlabel('VDD')
    plt.ylabel('Delay (s)')
    plt.title('Delay vs VDD')

def plot_W_n(model,Wn_range=Wn_range,VDD=1.8,C_L=0.5e-12):
    # Varying W_n
    W_n=np.linspace(*Wn_range,200)
    VDD=np.array(200*[VDD])
    C_L=np.array(200*[C_L])
    inputs=pd.DataFrame({'VDD':VDD,'W_n':W_n,'C_L':C_L})
    predictions=model.predict(np.array(inputs))
    delay,power=predictions[:,0],predictions[:,1]

    # Scatter plot for power
    plt.subplot(3,2,3)
    plt.plot(W_n, power)
    plt.xlabel('W_n (m)')
    plt.ylabel('Power (W)')
    plt.title('Power vs W_n')

    # Scatter plot for delay
    plt.subplot(3,2,4)
    plt.plot(W_n, delay)
    plt.xlabel('W_n (m)')
    plt.ylabel('Delay (s)')
    plt.title('Delay vs W_n')

def plot_C_L(model,CL_range=CL_range,VDD=1.8,W_n=1e-6):
    # Varying C_L
    C_L=np.linspace(*CL_range,200)
    W_n=np.array(200*[W_n])
    VDD=np.array(200*[VDD])
    inputs=pd.DataFrame({'VDD':VDD,'W_n':W_n,'C_L':C_L})
    predictions=model.predict(np.array(inputs))
    delay,power=predictions[:,0],predictions[:,1]

    # Scatter plot for power
    plt.subplot(3,2,5)
    plt.plot(C_L, power)
    plt.xlabel('C_L (F)')
    plt.ylabel('Power (W)')
    plt.title('Power vs C_L')

    # Scatter plot for delay
    plt.subplot(3,2,6)
    plt.plot(C_L, delay)
    plt.xlabel('C_L (F)')
    plt.ylabel('Delay (s)')
    plt.title('Delay vs C_L')
    
    plt.tight_layout()
    plt.show
            