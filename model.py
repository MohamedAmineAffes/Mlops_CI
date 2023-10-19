#!/usr/bin/env python
# coding: utf-8

# In[13]:


# Import modules and packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# In[21]:


# Functions and procedures
def plot_predictions(train_data, train_labels,  test_data, test_labels,  predictions):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(6, 5))
  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", label="Training data")
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", label="Testing data")
  # Plot the predictions in red (predictions were made on the test data)
  plt.scatter(test_data, predictions, c="r", label="Predictions")
  # Show the legend
  plt.legend(shadow='True')
  # Set grids
  plt.grid(which='major', c='#cccccc', linestyle='--', alpha=0.5)
  # Some text
  plt.title('Model Results', family='Arial', fontsize=14)
  plt.xlabel('X axis values', family='Arial', fontsize=11)
  plt.ylabel('Y axis values', family='Arial', fontsize=11)
  # Show
  plt.savefig('model_results.png', dpi=120)



def mae(y_test, y_pred):
  """
  Calculuates mean absolute error between y_test and y_preds.
  """
  return tf.metrics.mean_absolute_error(y_test, y_pred)


# In[22]:


# Check Tensorflow version
print(tf.__version__)


# In[23]:


# Create features
X = np.arange(-100, 100, 4)

# Create labels
y = np.arange(-90, 110, 4)


# Split data into train and test sets
X_train = X[:40] # first 40 examples (80% of data)
y_train = y[:40]

X_test = X[40:] # last 10 examples (20% of data)
y_test = y[40:]


# In[17]:


# Create features
X = np.arange(-100, 100, 4)

# Create labels
y = np.arange(-90, 110, 4)

# Split data into train and test sets
X_train = X[:40]  # first 40 examples (80% of data)
y_train = y[:40]

X_test = X[40:]   # last 10 examples (20% of data)
y_test = y[40:]

# Set random seed
tf.random.set_seed(42)

# Create a model using the Sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),  # Input layer with a shape of (1,) for one feature
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(loss=tf.keras.losses.mean_absolute_error,
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              metrics=['mae'])

# Fit the model
model.fit(X_train, y_train, epochs=100)  # Change epochs to 100, added verbose parameter
# Evaluate the model on the test set
mae = model.evaluate(X_test, y_test, verbose=0)
print(f'Mean Absolute Error on Test Data: {mae[1]:.2f}')
# Write metrics to file
with open('metrics.txt', 'w') as outfile:
    outfile.write(f'\nMean Absolute Error = {mae}.')


# In[20]:





# In[ ]:




