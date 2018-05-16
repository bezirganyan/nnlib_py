import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from NN.Classifier import Classifier
from NN.Losses import lse
from NN.layer import Layer
from NN import Optimizers

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encode the cathegorical data

X_labelEncoder = LabelEncoder()
X[:, 1] = X_labelEncoder.fit_transform(X[:, 1])
X[:, 2] = X_labelEncoder.fit_transform(X[:, 2])

X_oneHotEncoder = OneHotEncoder(categorical_features=[1])
X = X_oneHotEncoder.fit_transform(X).toarray()
X = X[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

optimizer = Optimizers.RMSProp()

classifier = Classifier(loss=lse, epochs=20, batch_size=32, history=True)
classifier.add_layer(Layer(neurons_num=6, activation='relu', input_dim=11, optimizer=optimizer))
classifier.add_layer(Layer(neurons_num=6, activation='relu', optimizer=optimizer))
classifier.add_layer(Layer(neurons_num=1, activation='sigmoid', optimizer=optimizer))

classifier.fit(X_train, X_test, y_train, y_test)
classifier.plot()
