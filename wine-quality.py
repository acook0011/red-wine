"""
    List of models to write:
        - Regression 
        - Good vs. Bad Classification (ex. set <7 good, >7 bad)
        - Exact Quality Classification? (possible classifications in dataset: 3, 4, 5, 6, 7, 8)
        - Neural network
"""
############## Classification Model ####################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os, sys, warnings
from sklearn import metrics
import tensorflow as tf
from tensorflow import Dense
from tensorflow import Sequential


def filter_warnings(stop_print=True):
    """ Used for ignoring ConvergenceWarning Spam that come from terminating on max iterations
        False: "default" = Print warnings in terminal
        True: "ignore" = Do not print warnings in terminal """
    if not stop_print:
        state = "default"
    else:
        state = "ignore"
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = state


def quality_classification(df):
    # Separate features (X) and target variable (y)
    y = df["quality"]
    X = df.drop("quality", axis=1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # predictions
    predictions = rf_classifier.predict(X_test)

    # Model evaluation
    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions, zero_division=1))


def regression_quality(df):
    df = df.replace({'quality': {
        8: 'Good', 7: 'Good',
        6: 'Average', 5: 'Average',
        4: 'Bad', 3: 'Bad', }})

    # Separate features (X) and target variable (y)
    y = df["quality"]
    X = df.drop("quality", axis=1)

    # Impute missing values with the mean before split
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # First split testing and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)

    # Create a Logistic Regression Model
    logi_reg_base = LogisticRegression()

    # Baseline Logistic Regression Model
    filter_warnings()
    logi_reg_base.fit(X_train, y_train)
    filter_warnings(False)

    # Baseline Training Set Predictions
    print("Training Set (Baseline)")
    base_train_predict = logi_reg_base.predict(X_train)
    print(classification_report(y_train, base_train_predict))

    # Baseline Testing Set Predictions
    print("Testing Set (Baseline)")
    base_test_predict = logi_reg_base.predict(X_test)
    print(classification_report(y_test, base_test_predict))

    # Tune Logistic Regression over a GridSearch with cross-validation
    param_grid = [
        {'C': [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
         'class_weight': ['balanced'],
         'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
        }
    ]
    logi_grid_opti = GridSearchCV(logi_reg_base, param_grid=param_grid, cv=5)
    filter_warnings()
    logi_grid_opti.fit(X_train, y_train)
    filter_warnings(False)

    # Print Best Parameters from GridSearch
    print(logi_grid_opti.best_params_)
    print(logi_grid_opti.best_score_)
    logi_reg = LogisticRegression(**logi_grid_opti.best_params_)
    logi_reg.fit(X_train, y_train)

    # Training Set Predictions
    print("Training Set")
    opti_train_predict = logi_reg.predict(X_train)
    print(classification_report(y_train, opti_train_predict))

    # Testing Set Predictions
    print("Testing Set")
    opti_test_predict = logi_reg.predict(X_test)
    print(classification_report(y_test, opti_test_predict))

    # Display Data on Chart (baseline and optimized)
    title = "Logistic Regression Quality: Weighted Averages"
    label_list = ["Base Training", "Base Testing", "Tuned Training", "Tuned Testing"]
    accuracies = [f1_score(y_train, base_train_predict, average="weighted"),
                  f1_score(y_test, base_test_predict, average="weighted"),
                  f1_score(y_train, opti_train_predict, average="weighted"),
                  f1_score(y_test, opti_test_predict, average="weighted")]
    difference = []
    labels = []
    for a, la in zip(accuracies, label_list):
        # Prints accuracy on chart
        a_percent = round(a*100, 2)
        labels.append(la + "\n" + str(a_percent) + "%")
        # Calculates difference for top portion of bar chart
        diff = 1 - a
        difference.append(diff)
    plt.bar(labels, accuracies, color="#80b870")
    plt.bar(labels, difference, bottom=accuracies, color="#f04f4d")
    plt.legend(["Correct", "Wrong"])
    plt.title(title)
    plt.show()


def good_vs_bad_model(df):
    # adding classifiers to each quality score beyond 7.
    df['quality'] = df['quality'].apply(lambda x: -1 if x < 7 else (1 if x == 7 else (2 if x == 8 else (3 if x == 9 else 4))))

    # Separate features (X) and target variable (y)
    X = df.drop('quality', axis=1)
    y = df['quality']

    # Impute missing values with the mean before split
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # First split testing and training sets
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Split validation set from full training set
    X_train_part, X_val, y_train_part, y_val = train_test_split(X_train_full, y_train_full, test_size=0.3, random_state=42)

    # Create a Decision Tree classifier
    clf = DecisionTreeClassifier(random_state=42)

    # Train the model on part of training set to compare to validation set
    clf.fit(X_train_part, y_train_part)

    # Print accuracy on the partial training set
    X_train_part_predict = clf.predict(X_train_part)
    train_accuracy = accuracy_score(y_train_part, X_train_part_predict)
    print(f"Training Accuracy: {train_accuracy:.2f}")
    print(classification_report(y_train_part, X_train_part_predict))

    # Print accuracy on the validation set
    X_val_predict = clf.predict(X_val)
    val_accuracy = accuracy_score(y_val, X_val_predict)
    print(f"Validation Accuracy: {val_accuracy:.2f}")
    print(classification_report(y_val, X_val_predict))

    # Train the model on full training set after finding optimal hyperparameters
    clf.fit(X_train_full, y_train_full)

    # Print accuracy on the test set
    X_test_predict = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, X_test_predict)
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print(classification_report(y_test, X_test_predict))

    # Make predictions on the entire dataset
    df['predicted_quality'] = clf.predict(imputer.transform(X))

    # Filter out rows with predicted quality equal to 2
    # df_filtered = df.loc[df['predicted_quality'] == 2]
    # df_filtered = df[df['predicted_quality'] >= 2]
    # df_filtered = df[df['predicted_quality'] != 1, df['predicted_quality'] != -1]

    # Sort the DataFrame based on predicted quality
    df_sorted = df.sort_values(by='predicted_quality', ascending=False)

    # Display the top 18 wines because only 18 wines have a quality score of 8
    top18_wines = df_sorted.head(18)[['wine number', 'quality', 'predicted_quality']]
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


############## Keras Neural Network ####################
def Neural_network(df):
    # Separate features (X) and target variable (y)
    y = df["quality"]
    X = df.drop("quality", axis=1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential()
    # Added the input layers for all sections of data
    model.add(Dense(12, activation='sigmoid', input_shape=(11,)))
    # Add hidden layers to model
    model.add(Dense(9, activation='relu'))
    # Only one output wanted once data is assessed
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
    Needed = model.fit(X_train, y_train,epochs=20, batch_size=2, verbose=2)
    print(Needed)

    #Now we need to predict the changes from this neural model
    y_pred = (model.predict(X_test))
    y_pred[0:10]

    accuracytrain = model.evaluate(X_test, y_test, verbose=0)
    print(accuracytrain, 'Train Accuracy: % acc')
    accuracytest = model.evaluate(X_train, y_train, verbose=0)
    print(accuracytest, 'Test Accuracy: % acc')

    print(pd.DataFrame(metrics.confusion_matrix(y_test, y_pred, labels=[0 ,1]), index=['true:White', 'true:Red'], columns=['pred:White', 'pred:Red']))

    # plot accuracy with training only to see if the accuracy is better
    plt.title('Accuracy for training')
    plt.plot(Needed.history['accuracy'], label='Training accuracy')
    plt.legend()


file_path = 'red wine data 1.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')
wine_df = pd.DataFrame(data)

# uncomment to run models
quality_classification(wine_df)
regression_quality(wine_df)
good_vs_bad_model(wine_df)
Neural_network(wine_df)

