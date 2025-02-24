# Spam Detection with Naive Bayes

This project is a simple implementation of a **Spam Detection System** using the Naive Bayes algorithm. The project is developed in Python and uses natural language processing (NLP) techniques to classify email/text messages as spam or not.

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Technologies Used](#technologies-used)
4. [Project Workflow](#project-workflow)
5. [Model Training](#model-training)
6. [Evaluation](#evaluation)
7. [Result](#result)

## Overview

This project utilizes **Multinomial Naive Bayes** to predict whether a given message is **spam** or **not spam**. The project involves text preprocessing techniques, feature extraction using **CountVectorizer** and **TF-IDF Vectorizer**, and then applying machine learning for classification.

The goal is to classify email/text messages and filter out spam.

## Dataset

The dataset used in this project contains two columns:
- **text**: The email or message content.
- **spam**: The target variable that contains binary values: `1` for spam and `0` for ham (not spam).

The dataset is loaded from a CSV file, and basic data cleaning techniques such as removing duplicates are applied.

## Technologies Used

- **Python** (Programming Language)
- **Pandas** (Data manipulation)
- **Numpy** (Numerical computing)
- **Matplotlib** & **Seaborn** (Data visualization)
- **NLTK** (Natural Language Toolkit for text preprocessing)
- **Scikit-learn** (Machine Learning)
- **CountVectorizer** & **TfidfVectorizer** (Text vectorization)
- **Naive Bayes Classifiers** (GaussianNB, MultinomialNB, BernoulliNB)

## Project Workflow

1. **Data Preprocessing**:
   - Removing duplicates.
   - Counting the number of sentences, words, and characters.
   - Plotting histograms and heatmaps for better data understanding.

2. **Text Preprocessing**:
   - Lowercasing.
   - Tokenization using `nltk`.
   - Removal of stopwords and punctuation.
   - Stemming using `PorterStemmer`.

3. **Vectorization**:
   - **CountVectorizer** is used to convert the transformed text into numerical features.

4. **Model Training**:
   - **Multinomial Naive Bayes** is applied to train the model on 80% of the dataset and tested on the remaining 20%.

## Model Training

Several Naive Bayes models were evaluated:
- **GaussianNB**
- **MultinomialNB**
- **BernoulliNB**

The dataset was split into training and test sets using an 80-20 ratio. Accuracy, confusion matrix, and precision metrics were calculated to evaluate the models.

## Evaluation

The performance of each model was evaluated based on the following metrics:
- **Accuracy**: The percentage of correctly classified instances.
- **Confusion Matrix**: A matrix to visualize the model's performance.
- **Precision**: The ability of the classifier not to label a negative sample as positive.
- **R2 Score**, **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)** for further evaluation.

## Results

After training the models, the **Multinomial Naive Bayes** performed best with an accuracy of approximately:

- **Accuracy**: 98.3%
- **Precision**: 96.8%
- **Confusion Matrix**: [values from your code]

You can check the bar plots for the most common words in spam and non-spam messages.
