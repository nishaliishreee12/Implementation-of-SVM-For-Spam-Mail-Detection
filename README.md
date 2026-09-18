# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:

To write a Python program to implement Support Vector Machine (SVM) for detecting whether an email or SMS message is spam or ham.

## EQUIPMENTS REQUIRED:

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter Notebook
3. spam.csv

## ALGORITHM

1. Import the required Python libraries such as Pandas and Scikit-learn.
2. Load the `spam.csv` dataset and select the message and label columns.
3. Convert the spam and ham labels into numerical values.
4. Split the dataset into training and testing data.
5. Convert the text messages into numerical feature vectors using TF-IDF Vectorization.
6. Create an SVM classifier using the Support Vector Machine algorithm.
7. Train the SVM model using the training messages and their corresponding labels.
8. Predict whether the messages in the testing dataset are spam or ham.
9. Calculate the accuracy of the SVM model.
10. Predict the class of a new message and display whether it is spam or ham.

## PROGRAM:

```python
# Program to implement SVM for Spam Mail Detection.
#
# Developed by:
# RegisterNumber:

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv(
    "spam.csv",
    encoding="latin1"
)

# Select only the required columns
data = data[["v1", "v2"]]

# Rename the columns
data.columns = ["Label", "Message"]

# Remove missing values
data = data.dropna()

# Convert labels into numerical values
# ham = 0
# spam = 1
data["Label"] = data["Label"].map({
    "ham": 0,
    "spam": 1
})

# Input and target
X = data["Message"]
y = data["Label"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert text into TF-IDF features
vectorizer = TfidfVectorizer(
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

# Create SVM classifier
model = SVC(
    kernel="linear"
)

# Train the SVM model
model.fit(
    X_train_tfidf,
    y_train
)

# Predict spam or ham
y_pred = model.predict(
    X_test_tfidf
)

# Calculate accuracy
accuracy = accuracy_score(
    y_test,
    y_pred
)

# Display actual and predicted values
print("Actual Labels:")
print(y_test.values[:20])

print("\nPredicted Labels:")
print(y_pred[:20])

# Display accuracy
print("\nAccuracy:", accuracy)

print(
    "\nAccuracy Percentage:",
    accuracy * 100,
    "%"
)

# Display confusion matrix
print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# Display classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Ham", "Spam"]
    )
)

# Test a new message
new_message = [
    "Congratulations! You have won a free lottery ticket. "
    "Click the link now to claim your prize."
]

# Convert the new message into TF-IDF features
new_message_tfidf = vectorizer.transform(
    new_message
)

# Predict the message
prediction = model.predict(
    new_message_tfidf
)

# Display prediction
print("\nNew Message:")
print(new_message[0])

if prediction[0] == 1:
    print("\nPredicted Result: Spam")
else:
    print("\nPredicted Result: Ham")
```

## Output:

<img width="597" height="423" alt="image" src="https://github.com/user-attachments/assets/275a7d31-2479-4f53-928a-49b39a7ab4be" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
