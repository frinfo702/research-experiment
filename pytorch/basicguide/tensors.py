# %%
import numpy as np
import torch

# %%
# Tensors are optimized for automatic differentiation

# Initializing a Tensor
# Tensors can be initialized in various ways

# Directly from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# From a Numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# From another tensor
x_ones = torch.ones_like(x_data)
print(f"Ones tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random tensor: \n {x_rand} \n")
# %%
shape = (2, 3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random tensor: \n {rand_tensor} \n")
print(f"Ones tensor: \n {ones_tensor} \n")
print(f"Zeros tensor: \n {zeros_tensor} \n")
# %%
# Attributes of a Tensor
# shape, datatype and the device on which they are stored

tensor = torch.rand(3, 4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device of tensor: {tensor.device}")
# %%
# Operations on Tensors
# By default, tensors are created on the CPU.
# Copying large tensors across devices can be expensive in terms of time and memory

if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())

# Standard numpy-like indexing and slicing
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")  # we can use ... instead of ":"
tensor[:, 1] = 0
print(tensor)
# %%
# Joining tensors
# Concatenate a sequence of tensors along a given dimension
# dim corresponds to direction of concatenation
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# %%
# @: matrix multiplication
# .T: returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

print(y1)

# This computes the element-wise product
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

print(z1)
