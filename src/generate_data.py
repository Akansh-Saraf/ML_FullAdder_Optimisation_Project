import numpy as np
import pandas as pd
from pyDOE import lhs

def main():
    # Define parameter ranges
    num_samples = 1000
    VDD_range = [1.5, 2.5]        # VDD: 1.5V to 2.5V
    Wn_range = [0.18e-6, 5e-6]     # W/L: 0.18µm to 5µm
    CL_range = [2e-14, 1e-12]     # C_L: 20fF to 1pF

    x=input_data(num_samples=num_samples,VDD_range=VDD_range,Wn_range=Wn_range,CL_range=CL_range)
    #x.to_csv('../data/Input_data_full_adder.csv')
    y=delay_data(num_samples=num_samples)
    df=power_data(num_samples=num_samples)
    y['Power']=df['Power']
    #y.to_csv('../data/Output_data_Full_Adder.csv')
    #plt.scatter(x['VDD'],y['Power'])

def input_data(num_samples = 1000,VDD_range = [1.5, 2.5],Wn_range = [0.1e-6, 5e-6],CL_range = [1e-14, 1e-12]):
    #Latin Hypercube Sampling
    lhs_samples = lhs(3, samples=num_samples)
    VDD_samples = lhs_samples[:, 0] * (VDD_range[1] - VDD_range[0]) + VDD_range[0]
    Wn_samples = lhs_samples[:, 1] * (Wn_range[1] - Wn_range[0]) + Wn_range[0]
    CL_samples = lhs_samples[:, 2] * (CL_range[1] - CL_range[0]) + CL_range[0]
    
    x = pd.DataFrame({
        'VDD': VDD_samples,
        'W_n': Wn_samples,
        'C_L': CL_samples  
    })
    
    with open("LTSpice/params.inc", "w") as file:
        # Write the .param for vdd
        file.write(".param vdd table(case")
        for i, value in enumerate(x['VDD'], start=1):
            file.write(f", {i},{value:.3f}")
        file.write(")\n")

        # Write the .param for W_n
        file.write(".param W_n table(case")
        for i, value in enumerate(x['W_n'], start=1):
            file.write(f", {i},{value:.3e}")
        file.write(")\n")

        # Write the .param for C_L
        file.write(".param C_L table(case")
        for i, value in enumerate(x['C_L'], start=1):
            file.write(f", {i},{value:.3e}")
        file.write(")\n")   
    return x

def power_data(num_samples=1000):
    with open("LTSpice/Full_Adder.log", "r") as file:
        for i,line in enumerate(file):
            if "avg_power" in line:
                skips_power=i

    y=pd.read_csv('LTSpice/Full_Adder.log',sep='\t',names=['step','Power','useless1','useless2'],skiprows=skips_power+2,nrows=num_samples,usecols=['Power'])
    #y.to_csv('Power_data_full_adder.csv')
    return y

def delay_data(num_samples=1000):
    with open("LTSpice/Full_Adder.log", "r") as file:
        for i,line in enumerate(file):
            if "t_p" in line:
                skips_delay=i

    y=pd.read_csv('LTSpice/Full_Adder.log',sep='\t',names=['step','Delay'],skiprows=skips_delay+2,nrows=num_samples,usecols=['Delay'])
    #y.to_csv('Delay_data_full_adder.csv')
    return y

if __name__=='__main__':
    main()