# import libraries
import os
import numpy as np
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle


# Helper functions to create dictionary and extract features from the corpus for model development
def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    all_words = []
    for email in emails:
        dirs = [os.path.join(email, f) for f in os.listdir(email)]
        for d in dirs:
            emails = [os.path.join(d, f) for f in os.listdir(d)]
            emails.sort()
            for mail in emails:
                with open(mail, errors="ignore") as m:
                    for i, line in enumerate(m):
                        if i == 2:  # Body of email is only 3rd line of text file
                            words = line.split()
                            all_words += words

    dictionary = Counter(all_words)
    # Code for non-word removal
    list_to_remove = list(dictionary.keys())
    for item in list_to_remove:
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000)
    return dictionary


def extract_features(mail_dir):
    files = [os.path.join(mail_dir, f) for f in os.listdir(mail_dir)]
    files.sort()
    docID = 0
    features_matrix = np.zeros((33716, 3000))
    train_labels = np.zeros(33716)
    for file in files:
        dirs = [os.path.join(file, f) for f in os.listdir(file)]
        for d in dirs:
            emails = [os.path.join(d, f) for f in os.listdir(d)]
            for mail in emails:
                with open(mail, errors="ignore") as m:
                    all_words = []
                    for line in m:
                        words = line.split()
                        all_words += words
                    for word in all_words:
                        wordID = 0
                        for i, d in enumerate(dictionary):
                            if d[0] == word:
                                wordID = i
                                features_matrix[docID, wordID] = all_words.count(word)
                train_labels[docID] = int(mail.split(".")[-2] == 'spam')
                docID = docID + 1
    return features_matrix, train_labels


# Use the above two functions to load dataset, create training, test splits for model development
train_dir = 'enron-spam'
dictionary = make_Dictionary(train_dir)

features, labels = extract_features(train_dir)

# Train SVM model
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.40)

model = LinearSVC()

model.fit(X_train, y_train)

# Test SVM model on unseen emails (test_matrix created from extract_features function)
result = model.predict(X_test)
print(confusion_matrix(y_test, result))
print(accuracy_score(y_test, result))

# Save your model as .sav file to upload with your submission
saved_model = "emailClassifer_enron.sav"

with open(saved_model, 'wb') as file:
    pickle.dump(model, file)






