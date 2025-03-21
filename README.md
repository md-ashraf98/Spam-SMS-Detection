# 📩 SMS Spam Detection Project

## 📚 Index
1. [Introduction](#Introduction)
2. [Requirements](#Requirements)
3. [Technologies-Used](#Technologies-Used)
4. [Working](#Working)
5. [Features](#Features)
6. [Conclusion](#Conclusion)



## 📖 Introduction

In today's world of communication, SMS remains one of the most widely used forms of messaging. However, the rise in spam messages has created challenges for both users and service providers. Spam messages can include promotional content, phishing scams, or fraudulent links that compromise user security.

The goal of this project is to build an automated **SMS Spam Classifier** using **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques. The classifier aims to differentiate spam messages from legitimate ones (ham) with high accuracy.

This project uses a labeled dataset containing SMS messages classified as either "spam" or "ham". We apply data preprocessing steps like text cleaning, stopwords removal, tokenization, and lemmatization to prepare the data. The **TF-IDF Vectorizer** is then used to convert textual data into numerical vectors, which serve as input to a **Logistic Regression** model for classification.



## ⚙️ Requirements

- Python 3 version
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- pickle

  

## 🖥️ Technologies-Used

- **Python 3.x** 🐍 - Primary programming language for model development.
- **Pandas** 📊 - For data manipulation and analysis.
- **NumPy** 🔢 - For numerical operations.
- **Matplotlib** & **Seaborn** 📈 - For data visualization and plotting.
- **NLTK (Natural Language Toolkit)** ✂️ - For text preprocessing (stopwords, lemmatization).
- **Scikit-learn (sklearn)** 🤖 - For vectorization (TF-IDF) and Logistic Regression modeling.
- **Pickle** 💾 - For saving and loading the model and vectorizer.
- **Google Colab** 📓 - For coding, testing, and visualization.



## 🛠️ Working

### 📌 Step-by-step Procedure:

Step-1. **Dataset Loading**
   - Import the SMS spam dataset (CSV format).
   - Inspect the dataset to understand its structure.

Step-2. **Data Cleaning**
   - Removed unnecessary columns.
   - Handled missing values (if any).
   - Renamed columns for easier reference.

Step-3. **Exploratory Data Analysis (EDA)**
   - Visualized the spam vs ham distribution using bar plots.
   - Generated word clouds to highlight frequent words in spam and ham messages.
   - Analyzed message lengths to detect any trends or insights.

Step-4. **Text Preprocessing**
   - Converted text to lowercase.
   - Removed punctuation, special characters, numbers.
   - Removed common stopwords (e.g., "the", "and", "is").
   - Applied lemmatization to reduce words to their root forms.

Step-5. **Text Vectorization**
   - Used **TF-IDF Vectorizer** to convert the processed text into numerical feature vectors.

Step-6. **Model Training**
   - Split the data into training and test sets (75%-25%).
   - Trained a **Logistic Regression** model on the training set.
   - Evaluated model performance on the test set (Accuracy, Precision, Recall, F1-Score).

Step-7. **Model Testing**
   - Tested the model with custom SMS messages to verify its classification ability.

Step-8. **Model Saving**
   - Saved both the trained model and TF-IDF vectorizer using the **pickle** library.

Step-9. **Model Usage**
   - Loaded the saved model and vectorizer for future predictions on new/unseen messages.



## 🌟 Features

- 🔄 **End-to-End Pipeline**: From raw text to accurate predictions.
- 🧹 **Advanced Text Preprocessing**: Cleaning, stopword removal, and lemmatization.
- 🧮 **Feature Engineering**: TF-IDF based feature extraction for better performance.
- 🧠 **Logistic Regression Model**: Lightweight and interpretable classification algorithm.
- 📊 **Visualizations**: Includes bar plots to make data insights clearer.
- 💾 **Model Persistence**: Trained model and vectorizer are saved and reusable.
- 💡 **Custom Testing**: Allows input of custom SMS messages for real-time spam detection.
- 🚀 **Scalable & Maintainable**: Easily extendable to other text classification tasks.



## 📝 Conclusion

This project demonstrates a complete approach to solving the **SMS Spam Detection** problem using **NLP** and **Machine Learning** techniques.

- The model effectively distinguishes between spam and legitimate (ham) messages.
- Achieved high performance using **Logistic Regression** combined with **TF-IDF** features.
- Extensive **EDA** helped to understand the nature of spam vs ham messages.
- **Text preprocessing pipeline** ensures better generalization and cleaner inputs for the model.

The project is ready to be deployed or integrated into real-world applications where automated SMS filtering is required. 🚀



