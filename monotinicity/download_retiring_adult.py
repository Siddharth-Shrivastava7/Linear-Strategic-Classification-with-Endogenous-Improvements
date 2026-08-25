from curses import raw

from folktables import ACSDataSource, ACSIncome 
import numpy as np
import pandas as pd

cache_path="retiring_adult_data.csv" 

# This downloads the raw ACS PUMS CSVs to ./data on first call,
# then caches them locally for subsequent runs.
data_source = ACSDataSource(
    survey_year='2018',
    horizon='1-Year',
    survey='person',
)

# Pick one or more US states; use list_states() behavior or just pass codes.
ca_data = data_source.get_data(states=["CA"], download=True)

# Convert the pandas DataFrame into numpy arrays using the task definition.
# features: (n, d) numpy array
# labels:   (n,)   boolean array  (PINCP > 50000)
# group:    (n,)   numpy array    (RAC1P by default)
features, labels, group = ACSIncome.df_to_numpy(ca_data)

print(features.shape, labels.shape, group.shape)  

# Save processed arrays as a single CSV
df = pd.DataFrame(features, columns=ACSIncome.features)
df["label"] = labels.astype(int)
df.to_csv(cache_path, index=False)

print(f"Data saved to {cache_path}")


