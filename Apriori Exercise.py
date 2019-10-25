# Apriori

#Import libraries
import pandas as pd
import numpy as py
import matplotlib.pyplot as plt

#Import 
dataset = pd.read_csv('marketbasket.csv')
transactions = []
for i in range(0, 1362):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 256)])

# Training apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)
print(results[0])