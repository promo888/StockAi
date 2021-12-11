from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

#Create a dummy input
dummy_input = np.arange(1, 10) * 10 #to include? future max values
# or add/remove the First element / 10 (if MIN decrease in new batches) for scaling ONLY
# or add/remove the last element * 10(if MAX increase in future) for scaling ONLY
#in such way there is no need to rescale full dataset for a future batches with MAXimum values
#the same applies for MIN (if MINimum found in new batches)

print("Dummy Input = ", dummy_input)

#Create a standardscaler instance and fit the data
scaler = StandardScaler() 
output = scaler.fit_transform(dummy_input.reshape(-1, 1))
print("Output =\n ", list(output))
print("Output's Mean = ", output.mean())
print("Output's Std Dev = ", output.std())
print("\nAfter Inverse Transforming = \n", list(scaler.inverse_transform(output)))

#https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/




