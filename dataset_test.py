import utils
from onion_dataset import OnionDataset
import os
import matplotlib.pyplot as plt
import numpy as np

onions = OnionDataset(csv_file="data.csv", data_dir="proc", root_dir=os.getcwd())

fig = plt.figure()

for i in range(100):
    k = np.random.randint(0, np.random.randint(0,len(onions)))
    sample = onions[k]

    print(k, sample['image_i'].shape, sample['image_f'].shape, sample['u'])
    utils.validation_image(sample['image_i'], sample['image_f'], sample['u'], ["display"])
