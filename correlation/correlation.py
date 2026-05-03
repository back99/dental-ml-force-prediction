import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
dpa_025 = pd.read_csv('dpa_025.csv')
dpa_05  = pd.read_csv('dpa_05.csv')

df = pd.concat([dpa_025, dpa_05], ignore_index=True)

targets = ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']

# 2. Correlation Matrix
corr = df[targets].corr()
print(corr)

# 3. Plot
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            xticklabels=['Fx','Fy','Fz','Tx','Ty','Tz'],
            yticklabels=['Fx','Fy','Fz','Tx','Ty','Tz'])
plt.title('Correlation Matrix — 6 Force Components (DPA)')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: correlation_matrix.png")
