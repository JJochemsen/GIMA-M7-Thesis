import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import geopandas as gpd
import pandas as pd
from numpy import corrcoef
from numpy import polyfit


#Read input file
df = gpd.read_file("C:/Thesis_analysis/analysis_file.shp")

x = np.array(df['wegdek_num'])
y = np.array(df['intens_cor'])


plt.scatter(x, y)

z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x,p(x),"r--")

plt.xlabel("Road surface type")
plt.ylabel("Traffic counts")

plt.show()


print (df['brid_int'].corr(df['ftw_int']))