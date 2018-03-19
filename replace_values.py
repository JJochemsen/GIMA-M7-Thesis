import geopandas as gpd
import numpy as np

df = gpd.read_file("C:/Thesis/Analyse/analyse_test.shp")

df["num_schoon"] = df["SCHOONHEID"]
df["num_wegdek"] = df["WEGDEKSRT"]

df["num_schoon"] = df["num_schoon"].replace(['ONBEKEND', ' '], np.nan)
df["num_schoon"] = df["num_schoon"].replace(['zeer lelijk'], '1')
df["num_schoon"] = df["num_schoon"].replace(['lelijk/saai'], '2')
df["num_schoon"] = df["num_schoon"].replace(['neutraal'], '3')
df["num_schoon"] = df["num_schoon"].replace(['mooi'], '4')
df["num_schoon"] = df["num_schoon"].replace(['schilderachtig'], '5')

df["num_wegdek"] = df["num_wegdek"].replace(['ONBEKEND', ' '], np.nan)
df["num_wegdek"] = df["num_wegdek"].replace(['onverhard'], '1')
df["num_wegdek"] = df["num_wegdek"].replace(['halfverhard', 'klinkers', 'overig (hout/kinderkopjes e.d.)'], '2')
df["num_wegdek"] = df["num_wegdek"].replace(['tegels', 'asfalt/beton'], '3 ')

df.to_file("C:/Thesis/Analyse/testresults.shp")