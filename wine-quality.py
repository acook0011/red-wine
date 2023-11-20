import numpy as np
import pandas as pd

""" 
    List of models to write:
        - Regression 
        - Good vs. Bad Classification (ex. set <7 good, >7 bad)
        - Exact Quality Classification? (possible classifications in dataset: 3, 4, 5, 6, 7, 8)

############## Classification Model ####################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

file_path = 'red wine data 1.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')
df = pd.DataFrame(data)

# adding classifiers to each quality score beyond 7.
df['quality'] = df['quality'].apply(lambda x: -1 if x < 7 else (1 if x == 7 else (2 if x == 8 else (3 if x == 9 else 4))))

# Separate features (X) and target variable (y)
X = df.drop('quality', axis=1)
y = df['quality']

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Further split the temporary data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)
X_test_imputed = imputer.transform(X_test)


# Create a Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the model on the training set
clf.fit(X_train_imputed, y_train)


# Print accuracy on the training set
train_accuracy = accuracy_score(y_train, clf.predict(X_train_imputed))
print(f"Training Accuracy: {train_accuracy:.2f}")

# Print accuracy on the validation set
val_accuracy = accuracy_score(y_val, clf.predict(X_val_imputed))
print(f"Validation Accuracy: {val_accuracy:.2f}")

# Print accuracy on the test set
test_accuracy = accuracy_score(y_test, clf.predict(X_test_imputed))
print(f"Test Accuracy: {test_accuracy:.2f}")

# Make predictions on the entire dataset
df['predicted_quality'] = clf.predict(imputer.transform(X))

# Filter out rows with predicted quality equal to 2
df_filtered = df[df['predicted_quality'] == 2]
# df_filtered = df[df['predicted_quality'] >= 2]
# df_filtered = df[df['predicted_quality'] != 1, df['predicted_quality'] != -1]

# Sort the DataFrame based on predicted quality
df_sorted = df_filtered.sort_values(by='predicted_quality', ascending=False)

# Display the top 18 wines because only 18 wines have a quality score of 8
top18_wines = df_sorted.head(18)[['wine number', 'quality']]
print("Top 18 Wines with Quality Score 2:")
print(top18_wines)

# Count the wines of each quality score
quality_counts = df['predicted_quality'].value_counts()

# Print the number of wines in each quality score
for score, count in quality_counts.items():
    print(f"Quality Score {score}: {count} wines")

# Plot the bar graph
plt.bar(quality_counts.index, quality_counts.values, color='pink')
plt.xlabel('Quality Score')
plt.ylabel('Number of Wines')
plt.title('Distribution of Wines Across Quality Scores')
plt.show()


"""
