import numpy as np 
import pandas as pd 
import os

path_D1NAMO = '/home/filippos/repositories/rnn_diabetes/D1NAMO/'

def main():
    dataset_path = path_D1NAMO +'diabetes_subset'
    patient_directories = [os.path.join(dataset_path, o) for o in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path,o))]
    
    for patient_path in patient_directories:
        data = pd.read_csv(patient_path+"/glucose.csv") 
        # print(data.head())
        print(data.loc[2])
        exit()

if __name__ == '__main__':
    main()