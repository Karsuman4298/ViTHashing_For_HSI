import os
import h5py
import numpy as np

os.makedirs('data/Houston', exist_ok=True)

# Generate tiny Houston13.mat to speed up the dummy test
with h5py.File('data/Houston/Houston13.mat', 'w') as f:
    f.create_dataset('ori_data', data=np.random.rand(48, 200, 200).astype(np.float32))

# Generate tiny Houston13_7gt.mat (but 200x200 to pass split logic)
with h5py.File('data/Houston/Houston13_7gt.mat', 'w') as f:
    labels = np.zeros((200, 200), dtype=np.int64)
    # Train area (x < 60)
    labels[10, 10] = 1; labels[20, 20] = 2; labels[30, 30] = 3
    # Query area (80 <= x < 130)
    labels[10, 90] = 1; labels[20, 100] = 2; labels[30, 110] = 3
    # DB area (x >= 150)
    labels[10, 160] = 1; labels[20, 170] = 2; labels[30, 180] = 3
    f.create_dataset('map', data=labels)

print("Tiny mock data generated successfully.")
