import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_excel('M:/CropPrediction.xlsx')

x = data.drop('label', axis=1)
y = data['label']

model = RandomForestClassifier(n_estimators=20, min_samples_leaf=4, max_depth=10, criterion='gini')
model.fit(x, y)

pickle.dump(model, open('model.pkl','wb'))

print(model.predict(np.array([90, 42, 43, 20.45, 82, 6, 202]).reshape(1, -1)))
