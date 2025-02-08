import numpy as np
import pandas as pd
import folium
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('Thong_tin_covid.csv')
print(df.head())