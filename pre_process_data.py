import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt


path_D1NAMO = '/home/filippos/repositories/rnn_diabetes/D1NAMO/'

n_x = 6
fast_insulin_action_time = 60*60*4
slow_insulin_action_time = 60*60*24

# x = [datetime, glucose, insulin_fast, insulin_slow, activity, insulin_rate]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def create_dataset():
    dataset_path = path_D1NAMO +'diabetes_subset'
    patient_directories = [os.path.join(dataset_path, o) for o in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path,o))]
    
    all_patient_data = []

    for patient_path in patient_directories:
        print(patient_path)
        patient_data_array = np.empty((0,n_x))

        data_glucose = pd.read_csv(patient_path + "/glucose.csv") 
        data_insulin = pd.read_csv(patient_path + "/insulin.csv") 
        data_food = pd.read_csv(patient_path + "/food.csv") 

        data_glucose_np = data_glucose.to_numpy()
        data_insulin_np = data_insulin.to_numpy()
        data_food_np = data_food.to_numpy()

        for i in range(data_glucose_np.shape[0]):
            date_time_i = np.datetime64(data_glucose_np[i,0] + 'T' + data_glucose_np[i,1]).astype("float")
            glucose_i = data_glucose_np[i,2]
            if data_glucose_np[i,3] == 'cgm':
                patient_data_array = np.append(patient_data_array , np.array([[date_time_i , glucose_i, 0.0 , 0.0, 0.0, 0.0 ]]) , axis=0)

        for i in range(data_insulin_np.shape[0]):
            date_time_i = np.datetime64(data_insulin_np[i,0] + 'T' + data_insulin_np[i,1]).astype("float")
            j = np.argmax(patient_data_array[:,0]>date_time_i)
            
            if j == 0:
                continue

            insulin_fast_i = data_insulin_np[i,2]
            insulin_slow_i = data_insulin_np[i,3]
            
            if j > 0:
                glucose_interp = 0.5*(patient_data_array[j-1,1]+patient_data_array[j,1])
            else:
                glucose_interp = patient_data_array[-1,1]

            if insulin_fast_i > 0:
                pass
            else:
                insulin_fast_i = 0.0
            
            if insulin_slow_i > 0:
                pass
            else:
                insulin_slow_i = 0.0

            x_i = np.array([[date_time_i , glucose_interp, insulin_fast_i, insulin_slow_i, 0.0, 0.0]]) 

            patient_data_array = np.insert(patient_data_array , j , x_i , axis=0)

        #Add insulin rate data
        for i in range(patient_data_array.shape[0]):
            insulin_fast_i =  patient_data_array[i,2]
            insulin_slow_i =  patient_data_array[i,3]
            if insulin_fast_i > 0.0:
                date_time_i = patient_data_array[i,0]
                insulin_indices_i = np.logical_and(patient_data_array[:,0]>date_time_i,patient_data_array[:,0] < date_time_i + fast_insulin_action_time)
                patient_data_array[insulin_indices_i,5] += insulin_fast_i/fast_insulin_action_time

            if insulin_slow_i > 0.0:
                date_time_i = patient_data_array[i,0]
                insulin_indices_i = np.logical_and(patient_data_array[:,0]>date_time_i,patient_data_array[:,0] < date_time_i + slow_insulin_action_time)
                patient_data_array[insulin_indices_i,5] += insulin_slow_i/slow_insulin_action_time

        # Add activity data
        sensor_path = patient_path +'/sensor_data'
        sensor_directories = [os.path.join(sensor_path, o) for o in os.listdir(sensor_path) if os.path.isdir(os.path.join(sensor_path,o))]
        
        for i, sensor in enumerate(sorted(sensor_directories)):
            data_sensor = pd.read_csv(sensor + sensor[-20:] + "_Summary.csv") 
            if i == 0:
                data_sensor_np = data_sensor.to_numpy()
            else:
                data_sensor_np = np.concatenate((data_sensor_np,data_sensor.to_numpy())) 

        n_sensor = data_sensor_np.shape[0]
        for i in range(n_sensor):
            data_sensor_np[i,0] = 0.001*np.datetime64(data_sensor_np[i,0][6:10]+'-'+data_sensor_np[i,0][3:5]+'-'+data_sensor_np[i,0][0:2] + 'T'+ data_sensor_np[i,0][11:]).astype("float")

        for i in range(patient_data_array.shape[0]):
            date_time_i = patient_data_array[i,0]
            idx = (np.abs(data_sensor_np[:,0] - date_time_i)).argmin()
            patient_data_array[i,4] = np.mean(data_sensor_np[max(idx-2000,0):min(idx+2000,n_sensor),5])

        i_0_last_block = 0
        for i in range(patient_data_array.shape[0]-1):
            dt = patient_data_array[i+1,0] - patient_data_array[i,0]
            if dt > 900: # 15 minutes
                if i-i_0_last_block > 100:
                    all_patient_data.append(patient_data_array[i_0_last_block:i,:])
                    print('added block of length'+ str(i-i_0_last_block))

                i_0_last_block = i+1

        if i-i_0_last_block > 100:
            all_patient_data.append(patient_data_array[i_0_last_block:i,:])
            print('added block of length '+ str(i-i_0_last_block))

        #plotting of data for an individual
        plt.subplot(411)
        plt.plot((patient_data_array[:,0]-patient_data_array[0,0])/3600,patient_data_array[:,1])
        plt.ylabel('BG (mmol/L)')
        plt.gca().axes.get_xaxis().set_visible(False)       
        
        plt.subplot(412)
        plt.plot((patient_data_array[:,0]-patient_data_array[0,0])/3600,patient_data_array[:,2])
        plt.plot((patient_data_array[:,0]-patient_data_array[0,0])/3600,patient_data_array[:,3])
        plt.ylabel('Insulin Dose(U)')
        plt.legend(['Rapid Insulin', 'Long Lasting Insulin'])
        plt.gca().axes.get_xaxis().set_visible(False)       

        plt.subplot(413)
        plt.plot((patient_data_array[:,0]-patient_data_array[0,0])/3600,patient_data_array[:,5]*3600)
        plt.ylabel('Insulin Rate (U/hr)')
        plt.gca().axes.get_xaxis().set_visible(False)       

        plt.subplot(414)
        plt.plot((patient_data_array[:,0]-patient_data_array[0,0])/3600,patient_data_array[:,4])
        plt.xlabel('Time (hrs)')
        plt.ylabel('Activity Level')
        plt.show()


    # save data as .npz file
    np.savez('d1namo_insulin_rate', data =np.array(all_patient_data))

if __name__ == '__main__':
    create_dataset()