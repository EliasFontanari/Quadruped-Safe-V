import numpy as np
import matplotlib.pyplot as plt
import configuration.params as params

data_pairs = np.load('observation_datasets/pairs_aggregated.npy', mmap_mode='r')

print(f'Data loaded')
x = data_pairs[:,0]
y = data_pairs[:,1]

plt.figure()
plt.hist2d(x, y, bins=50, cmap='viridis', density='True',range=[[-params.x_lim, params.x_lim], [-params.y_lim, params.y_lim]])
plt.colorbar(label='Relative frequency')
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D Histogram of First Two Columns')
plt.tight_layout()
plt.show()

