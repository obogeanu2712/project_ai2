from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
# Fetch the dataset from UCI Machine Learning Repository

dataset = fetch_ucirepo(id=82)

data = dataset.data # from here you can access the data with .features and .target

features = data.features

# print(features.head())

#check for the entries that have missing values

null_features = (features.loc[features.isnull().any(axis=1)]) # we have 3 entries with missing values for 'COMFORT' feature

null_features.to_csv('null_features.csv', index=True)

mean_comfort = features['COMFORT'].mean()

print("mean comfort:", mean_comfort)

features.loc[features['COMFORT'].isnull(), 'COMFORT'] = mean_comfort

features.to_csv('processed_features.csv', index=False)

targets = data.targets

# print(targets.head())

df = pd.concat([features, targets], axis='columns')

# save the dataset to a CSV file
df.to_csv('dataset.csv', index=False)

# i noticed on line 4 'ADM-DECS' feature has a  'A ' value instaed of 'A'

targets.loc[targets['ADM-DECS'] == 'A ', 'ADM-DECS'] = 'A'


# features_names = features.columns.tolist()

# print("Features:", features_names.__str__())

features_columns =  ['L-CORE', 'L-SURF', 'L-O2', 'L-BP', 'SURF-STBL', 'CORE-STBL', 'BP-STBL', 'COMFORT']

# targerts_names = targets.columns.tolist()

# print("Targets:", targerts_names.__str__())

targets_columns = ['ADM-DECS']

ordinal_columns = ['L-CORE', 'L-SURF', 'L-O2', 'L-BP', 'SURF-STBL', 'CORE-STBL', 'BP-STBL']

numerical_columns = ['COMFORT']

categorical_columns = ['ADM-DECS']


# print(features.loc[:, ordinal_columns].head())

encoded = pd.DataFrame()

for column in ordinal_columns:
    enc = OrdinalEncoder()
    encoded.loc[:, [column]] = enc.fit_transform(features.loc[:, [column]])

# test : dont include columns with missing values -- concluded -- not fitted for 'COMFORT' column

# scaleing comfort column

scaler = StandardScaler()

encoded.loc[:, ['COMFORT']] = scaler.fit_transform(features.loc[:, ['COMFORT']])

enc = LabelEncoder()

encoded.loc[:, ['ADM-DECS']] = enc.fit_transform(targets.loc[:, ['ADM-DECS']].to_numpy().ravel())

encoded.to_csv('encoded.csv', index=False)

X = encoded.loc[:, ordinal_columns].values

y = encoded.loc[:, ['ADM-DECS']].values.ravel()

# training

first_layer = 100

layers = [(first_layer, 2 * first_layer), (first_layer, first_layer), (first_layer, int(first_layer / 2))]


learning_rates = [0.1, 0.01]

results = {}

for layer in layers:
    for learning_rate in learning_rates:
        mlp = MLPClassifier(hidden_layer_sizes=layer, learning_rate='constant', learning_rate_init=learning_rate, random_state=42, max_iter=2000)
        cv = cross_val_score(mlp, X, y, cv=4, scoring='accuracy')
        results[(layer, learning_rate)] = cv.mean()

    # Convert results to a DataFrame
    results_df = pd.DataFrame(
        [(layer, lr, acc) for (layer, lr), acc in results.items()],
        columns=["Layer", "Learning Rate", "Accuracy"]
    )

    results_df.to_csv('results.csv', index=False)