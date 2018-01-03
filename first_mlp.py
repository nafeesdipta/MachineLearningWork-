
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
# load dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=500, batch_size=10)
# Model Evaluation
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)
new_input = numpy.array([[3,88,58,11,54,24.8,267,22],
                         [6,92,92,0,0,19.9,188,28],
                         [10,101,76,48,180,32.9,171,63],
                         [2,122,70,27,0,36.8,0.34,27],
                         [5,121,72,23,112,26.2,245,30]])

predictions = model.predict(new_input)
print predictions # [1.0, 1.0, 1.0, 0.0(.36), 1.0] {3 layers}
                # # [1.0, 1.0, 1.0, 0.0(.40), 1.0] {4 layers}
