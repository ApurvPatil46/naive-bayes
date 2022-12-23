import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.model_section import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

dataset = pd.read_csv("cancer.csv")
# print(dataset.head())
dataset.info()

dataset = dataset.drop(["id"], axis=1)
dataset = dataset.drop(["Unnamed: 32"], axis=1)

# Maligant tumor dataframe
M = dataset[dataset.diagnosis == "M"]

# Benign tumor dataframe
B = dataset[dataset.diagnosis == "B"]

# We shall now examine malignant and benign tumors by examining their average radius and texture.
plt.title("Malignant vs Benign Tumor")
plt.xlabel("Radius Mean")
plt.ylabel("Texture Mean")
plt.scatter(M.radius_mean, M.texture_mean, color = "red", label = "Malignant", alpha = 0.3)
plt.scatter(B.radius_mean, B.texture_mean, color = "lime", label = "Benign", alpha = 0.3)
plt.legend()
plt.show()

# Processing
# Now, malignant tumors will be assigned a value of ‘1’ and benign tumors will be assigned a value of ‘0’.
dataset.diagnosis = [1 if i == "M" else 0 for i in dataset.diagnosis]

# We now divide our dataframe into x and y components. The x variable includes all independent predictor factors,
# whereas the y variable provides the diagnostic prediction.
x = dataset.drop(["diagnosis"], axis=1)
y = dataset.diagnosis.values

# Data Normalization
# To maximize the model’s efficiency, it’s always a good idea to normalize the data to a common scale.
x = (x - np.min(x)) / (np.max(x) - np.min(x))

# Test train split
# After that, we’ll use the train test split module from the sklearn package to divide the dataset into
# training and testing sections.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

# Now we’ll import and instantiate the Gaussian Naive Bayes module from SKlearn GaussianNB. To fit the model,
# we may pass x_train and y_train.
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)

# The following accuracy score reflects how successfully our Sklearn Gaussian Naive Bayes model predicted cancer
# using the test data.
print("Naive Bayes score: ",nb.score(x_test, y_test))

# Naive Bayes score:  0.935672514619883



