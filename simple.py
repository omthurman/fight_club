




import tensorflow as tf
print("Is TensorFlow using GPU? :", tf.test.is_gpu_available())

import torch
print("Is PyTorch using GPU? :", torch.cuda.is_available())


import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

x = torch.randn(3, 3).to(device)
y = torch.randn(3, 3).to(device)
z = x.mm(y)

print(z)


tf.config.list_physical_devices('GPU')


torch.cuda.is_available()



import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))


import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices())