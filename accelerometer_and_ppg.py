import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, find_peaks

# Go through the datasets one-by-one in a loop and create dummy accelerometer signal
def generate_dummy_accel(length_of_ppg):
    uniform_noise = np.random.rand(length_of_ppg, 3)
    acc = np.hstack((np.ones((length_of_ppg,2)),np.ones((length_of_ppg,1))*1000)) + uniform_noise;
    return acc

# Load PPG data from CSV
csv_file_path = "c:/Users/DYlmaz/Desktop/FM Based Modulation/participant_53.hdf5ppg.csv"
data = pd.read_csv(csv_file_path, skiprows=7500, nrows=10000)  # 400 seconds

# Step 2: Identify the data columns for the plot
y_column_index = 1  # Adjust the index based on CSV file
ppg_signal = data.iloc[:, y_column_index].to_numpy()

fs = 25.0  # Sample rate (Hz)
length_of_ppg_signal = len(ppg_signal)                  
accelerometer_signal= generate_dummy_accel(length_of_ppg_signal)
# now we have ppg signal with accelerometer data
combined_signals = np.hstack((ppg_signal.reshape(-1,1),accelerometer_signal))       

# Create a DataFrame from the combined_signals array
combined_df = pd.DataFrame(combined_signals, columns=['ppg_signal', 'x','y','z'])

# Define the output CSV file path
output_csv_path = "c:/Users/DYlmaz/Desktop/FM Based Modulation"
output_csv_name = 'combined_signals.csv'
full_output_name = os.path.join(output_csv_path,output_csv_name)

ibi_filename = 'ibi_result.csv'
full_ibi_name = os.path.join(output_csv_path,ibi_filename)

# Write the DataFrame to a CSV file
combined_df.to_csv(full_output_name, index=False)
#mxm_whrm_sample_code_continious_mode.exe this file creates ibi values from ppg signal and its corresponding x,y,z data
myCommand = '"c:\\Users\\DYlmaz\\Desktop\\FM Based Modulation\\mxm_whrm_sample_code_continious_mode.exe ' + full_output_name + ' > ' + full_ibi_name 
os.system(myCommand)

df = pd.read_csv(full_ibi_name)
df.columns = ['HR','HR_Confidence','IBI','IBI_Confidence']
print('df.columns: ', df.columns)
df.to_csv(full_ibi_name, index=False, header = True)

df = pd.read_csv(full_ibi_name)

# Extract IBI values from the DataFrame
ibi_values_with_zero = df['IBI'].to_numpy()
# Remove 0 values from ibi_values
ibi_values = ibi_values_with_zero[ibi_values_with_zero != 0]
cumsum_ibis = np.cumsum(ibi_values)/1000

# Create a stem plot of IBI values
plt.figure(figsize=(10, 6))
plt.stem(cumsum_ibis, ibi_values, linefmt='b-')
plt.xlabel('Cumulative Sum of IBI(s)')
plt.ylabel('Interbeat Interval (ms)')
plt.title('Interbeat Interval (IBI) Stem Plot')
plt.grid(True)
plt.savefig("IBI.png")
plt.show()

# Apply low-pass filter
def butter_lowpass(cutoff, fs, order=5):  
    normal_cutoff = cutoff / 0.5 * fs
    print('normal_cutoff: ',normal_cutoff)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Define the filter parameters
cutoff_frequency = 0.1
sampling_frequency = 1000 / np.mean(ibi_values) #400 / len(ibi_values) 
print('mean: ',np.mean(ibi_values) )
print('sampling frequency: ',sampling_frequency)
filtered_ibi_values = butter_lowpass_filter(ibi_values, cutoff_frequency, sampling_frequency)

# Detect peaks in the filtered IBI values
peaks, _ = find_peaks(filtered_ibi_values, height=1)

# Calculate time intervals between peaks in milliseconds
peak_times = peaks / sampling_frequency  * 1000  # Convert peak indices to milliseconds
time_intervals_ms = np.diff(peak_times)  # Calculate time intervals between peaks in milliseconds

peak_times_seconds = peak_times / 1000

# Calculate respiration rate in breaths per minute (BPM) for each millisecond
respiration_rates = 60 * 1000 / time_intervals_ms

print('respiration_rates: ', respiration_rates)

# Plot the original and filtered IBI values with detected peaks
plt.figure(figsize=(10, 6))
plt.plot(cumsum_ibis, ibi_values, label='Original IBI')
plt.plot(cumsum_ibis, filtered_ibi_values, label='Filtered IBI')
plt.plot(cumsum_ibis[peaks], filtered_ibi_values[peaks], 'ro', label='Detected Peaks')  # Mark detected peaks with red circles
plt.xlabel('Cumulative Sum of IBI(s)')
plt.ylabel('Interbeat Interval (ms)')
plt.title('Original vs. Filtered Interbeat Interval (IBI) with Detected Peaks')
plt.legend()
plt.grid(True)
plt.savefig("Original_AND_Filtered_IBI.png")
plt.show()

# Plot respiration rate over time with markers
plt.figure(figsize=(10, 6))
plt.plot(peak_times_seconds[1:], respiration_rates, label='Respiration Rate', marker='o', linestyle='-', markersize=5)  # 'o' marker style
plt.xlabel('Time (seconds)')
plt.ylabel('Respiration Rate (breaths per minute)')
plt.legend()
plt.title('Respiration Rate Over Time')
plt.savefig("FM_Based_Modulation_RR.png")
plt.show()








