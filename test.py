# import pandas as pd

# d = {'col1': [1, 2, 3, 4, 5],
#                    'col2': ['A', 'B', 'C', 'D', 'E'],
#                    'col3': [10.5, 20.5, 30.5, 40.5, 2]}

# df = pd.DataFrame(data=d)

# slice_df = df.loc[:, ['col1', 'col3']]
# print(slice_df)

# from sklearn.neural_network import MLPClassifier
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# cancer_data = load_breast_cancer()
# X, y = cancer_data.data, cancer_data.target

# print(X.shape, y.shape)


import pandas as pd

df = pd.DataFrame(data={'col1': [1, 2, 3, 4, 5],
                   'col2': ['A', 'B', 'C', 'D', 'E'],
                   'col3': [10.5, 20.5, 30.5, 40.5, 3]})




print(df['col1'].mean())

