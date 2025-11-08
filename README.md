# Movie Review Sentiment Analysis with NLTK

This repository contains a Python Jupyter Notebook that demonstrates how to perform sentiment analysis on movie reviews using the Natural Language Toolkit (NLTK) library. The project utilizes a Naive Bayes classifier to distinguish between positive and negative reviews based on their textual content.

## Project Overview

The goal of this project is to build and evaluate a sentiment analysis model that can automatically classify movie reviews as either positive or negative. This is achieved by training a Naive Bayes classifier on a labeled dataset of movie reviews. The notebook walks through the entire process, from data loading and preprocessing to model training and evaluation.

### Key Features:
*   **Data Loading and Exploration:** The project uses the `movie_reviews` dataset available in NLTK, which contains 2000 movie reviews categorized as "positive" or "negative".
*   **Text Preprocessing:** The notebook demonstrates essential text preprocessing techniques, including:
    *   **Tokenization:** Breaking down the text into individual words or tokens.
    *   **Stopword Removal:** Eliminating common words (e.g., "the," "a," "is") that do not contribute much to the overall sentiment.
    *   **Punctuation Removal:** Stripping out punctuation to focus on the meaningful words.
*   **Feature Extraction:** The preprocessed words are converted into a feature set suitable for the machine learning model.
*   **Model Training:** A Naive Bayes classifier is trained on the prepared feature set from the movie reviews.
*   **Model Evaluation:** The performance of the trained classifier is evaluated to determine its accuracy in classifying unseen reviews.
*   **Informative Feature Analysis:** The notebook identifies and displays the most informative words that the classifier uses to distinguish between positive and negative reviews.

## Getting Started

To run this project, you will need to have Python and the NLTK library installed.

### Prerequisites

*   Python 3.x
*   NLTK: `pip install nltk`

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd <repository-name>
    ```
3.  **Download the NLTK data:**
    Open a Python interpreter and run the following commands to download the necessary NLTK datasets, including the movie reviews corpus and stopwords.

    ```python
    import nltk
    nltk.download('movie_reviews')
    nltk.download('stopwords')
    nltk.download('punkt')
    ```

### Usage

1.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
2.  **Open the notebook:**
    Open the `Tokenizing.ipynb` file to view and run the code.

## Project Workflow

The notebook is structured to guide you through the following steps:

1.  **Setting up the Environment:** The first step involves importing the NLTK library and downloading all the necessary components.
2.  **Tokenization:** The process of breaking down a stream of text into words, phrases, symbols, or other meaningful elements called tokens is demonstrated.
3.  **Stopword Removal:** The notebook shows how to remove common English stopwords from the tokenized text. NLTK's pre-defined list of stopwords is used for this purpose.
4.  **Loading the Movie Reviews Dataset:** The `movie_reviews` corpus from NLTK is loaded. This dataset contains 1000 positive and 1000 negative movie reviews.
5.  **Data Cleansing:** The reviews are cleaned by removing stopwords and punctuation.
6.  **Frequency Distribution:** The frequency of words in both positive and negative reviews is analyzed and plotted to gain insights into the vocabulary of each category.
7.  **Building the Sentiment Analysis Classifier:**
    *   A feature extractor is created to prepare the data for the Naive Bayes classifier.
    *   The dataset is split into training and testing sets.
    *   The Naive Bayes classifier is trained on the training set.
8.  **Evaluating the Classifier:** The accuracy of the classifier is calculated using the test set.
9.  **Analyzing Informative Features:** The notebook concludes by displaying the words that are most influential in determining whether a review is positive or negative.

## Results

The Naive Bayes classifier is trained and tested on the movie review dataset. The notebook provides an accuracy score for the model and lists the most informative features. For instance, words like "outstanding" and "captures" are strong indicators of a positive review, while words like "seagal" and "ludicrous" are associated with negative reviews.

The project also includes an example of how to classify a custom movie review using the trained model, demonstrating its practical application.
