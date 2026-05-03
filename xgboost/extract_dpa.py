import pandas as pd

# Load Smith dataset
df = pd.read_csv('smith_dataset.csv')

# Extract DPA only
dpa = df[df['type'] == 'DPA'].copy()

# Split by activation level
dpa_025 = dpa[dpa['thickness'] == 0.25].copy()
dpa_05  = dpa[dpa['thickness'] == 0.5].copy()

print(f"DPA 0.25: {dpa_025.shape}")
print(f"DPA 0.5 : {dpa_05.shape}")

# Save
dpa_025.to_csv('dpa_025.csv', index=False)
dpa_05.to_csv('dpa_05.csv',  index=False)

print("Done! → dpa_025.csv, dpa_05.csv")
