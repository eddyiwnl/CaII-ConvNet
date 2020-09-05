import h5py
import numpy as np


print('available models: alex, revised, resnet, inception, VGG')
model_name = input('Select a model from the above list: ')

result_name=model_name+'_results.hdf5'
with h5py.File(result_name,'r') as hf:
    results = hf["results"][:]


print("acc")
print(results[0:5,1])
print("precision")
print(results[0:5,2])
print("recall")
print(results[0:5,3])
print("f1")
print(results[0:5,4])
print("roc")
print(results[0:5,5])


print("mean acc")
print(np.mean(results[0:5,1]) )
print("mean precision")
print(np.mean(results[0:5,2]) )
print("mean recall")
print(np.mean(results[0:5,3]) )
print("mean f1")
print(np.mean(results[0:5,4]) )
print("mean roc")
print(np.mean(results[0:5,5]) )

