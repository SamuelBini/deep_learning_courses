import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_sample_image

# Chargement des images
china = load_sample_image("china.jpg") / 255
flower = load_sample_image("flower.jpg") / 255
dataset = np.array([china, flower], dtype=np.float32)
batch_size, height, width, channels = dataset.shape


# Création de deux filtres



plt.imshow(china)
plt.show()