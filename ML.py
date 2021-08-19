# Importing Libraries
import tensorflow as tf
import matplotlib.pyplot as plt #for visualisation
import numpy as np
from sklearn.model_selection import StratifiedKFold #for cross-validation

# Building Model
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# convert the feature values from integers to floating-point numbers from 0 to 1
x_train, x_test = x_train / 255.0, x_test / 255.0

print("The variables x_train and y_train represent the features (x) and targets (y) of the training set.")
print()
print("The shape of x_train is {}.".format(x_train.shape))
print("This means that there are {} instances in the training set.".format(x_train.shape[0]))
print("Each instance is a {} by {} grid of black-and-white pixels, with values 0 to 1, depending on the intensity ('blackness') of the pixel.".format(x_train.shape[1],x_train.shape[2]))
print()
print("The shape of y_train is {}.".format(y_train.shape))
print("This simply means that there are single target values for each of the {} instances".format(y_train.shape[0]))
print("Each target value is a digit from '0' to '9', indicating what digit the feature-pixels represent.")
print()
print("The variables x_test and y_test mean the same as x_train and y_train, except this is the test set instead of the training set.")
print("There are {} instances in the test set.".format(x_test.shape[0]))

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
print("The prediction logits for the first training example is: ")
print(predictions)

tf.nn.softmax(predictions).numpy()

# Calculating Loss
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
print("The loss for the first training example is: ")
print(loss_fn(y_train[:1], predictions).numpy())

# Training Model
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

# Testing Model
model.evaluate(x_test,  y_test, verbose=2)
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

ind2 = int(input("Enter the index of the test digit you wish to see (0 to 9999): "))

# Display the target and prediction of digit
trained_predictions = probability_model.predict(x_test)
pred = np.where(trained_predictions[ind2]==max(trained_predictions[ind2]))[0][0]
print("Digit at index {} has prediction probabilities: {}.".format(ind2,trained_predictions[ind2]))
print("Digit at index {} is labelled as '{}'.".format(ind2, y_test[ind2]))
print("Digit at index {} is predicted to be '{}'.".format(ind2,pred))

# Cross Validation
k = 5 # The number of folds.
skf = StratifiedKFold(n_splits=k, shuffle=True)

for i, (train, eval) in enumerate(skf.split(x_train, y_train)):
    print("Running Fold", i+1, "/", k)
    # Create the model
    xmodel = model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
        ])
    # Compile the model
    xmodel.compile(optimizer='adam',
        loss=loss_fn,
        metrics=['accuracy'])
    # Fit the model
    xmodel.fit(x_train[train], y_train[train], epochs=5*i)
    # Train the model
    print("Evaluation on Validation set: ")
    xmodel.evaluate(x_train[eval], y_train[eval], verbose=2)
    print()