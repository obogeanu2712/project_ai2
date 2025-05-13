import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=82)

targets = dataset.data.targets

print(targets.head())

# make a histogram of the target variable 'ADM-DECS'

targets.loc[targets['ADM-DECS'] == 'A ', 'ADM-DECS'] = 'A'

values = targets['ADM-DECS'].to_numpy()

print(values)

#calculate the frequency of each unique value in the target variable
value_counts = pd.Series(values).value_counts()
print(f'{(value_counts / values.size).round(2)}')

# Create a histogram of the target variable 'ADM-DECS'
plt.hist(values, bins=10, edgecolor='black', alpha=0.7)
plt.title('Histogram of ADM-DECS')
plt.xlabel('ADM-DECS')
plt.ylabel('Frequency')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(axis='y', alpha=0.75)  # Add grid lines for better readability
plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels



plt.show()