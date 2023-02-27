# tensorflow practice

import pandas as pd
from sklearn.model_selection import train_test_split
# importing tensorflow
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score


# getting dataset
df = pd.read_csv('datasets/Churn.csv')

# getting X and y from dataset
X = pd.get_dummies(df.drop(['Churn', 'Customer ID'], axis=1))
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# splitting into training and testing groups
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2)


# building the model
model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=len(X_train.columns)))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# compiling model
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')

#fitting model
model.fit(X_train, y_train, epochs=200, batch_size=32)


# lets us predict from model
y_hat = model.predict(X_test)
y_hat = [0 if val < 0.5 else 1 for val in y_hat]

# prediction accuracy
print(accuracy_score(y_test, y_hat))


model.save('tfmodel')











