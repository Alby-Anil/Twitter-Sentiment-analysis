# Twitter Sentiment Analysis

This project performs sentiment analysis on Twitter data using the Sentiment140 dataset. It demonstrates an end-to-end natural language processing (NLP) pipeline including data preprocessing, feature extraction (TF-IDF), and classical machine learning modeling with Logistic Regression.

---

## Project Overview

- **Goal:** Classify tweets as positive or negative sentiment from real-world social media text.
- **Dataset:** Sentiment140, a dataset with 1.6 million annotated tweets.
- **Methodology:**  
  - Text cleaning and preprocessing (removing URLs, mentions, punctuation, stopwords, stemming).  
  - Feature extraction using TF-IDF vectorization.  
  - Training a Logistic Regression classifier for sentiment detection.  
  - Model evaluation with classification reports and confusion matrices.  
  - Interpretation of top sentiment words and visualization with word clouds.

---

## Repository Structure

twitter-sentiment-analysis/
│
├── Data/ # Dataset CSV (not included due to size)
│ └── training.1600000.processed.noemoticon.csv (download separately)
├── Models/ # Saved ML model & vectorizer
│ ├── logreg_sentiment140.pkl
│ └── tfidf_vectorizer.pkl
├── Notebooks/ # Jupyter notebook(s) with all code
│ └── sentiment_analysis.ipynb
├── .gitignore # Git ignore file
└── README.md # Project overview (this file)


---

## Dataset Information

- The original Sentiment140 dataset file (`training.1600000.processed.noemoticon.csv`) is **not included** in this repository due to GitHub’s 100 MB file size limit.
- You can download it here: [Sentiment140 on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)
- After downloading, place the dataset CSV inside the `Data/` directory to run the notebook successfully.

---

## Usage Instructions

1. Clone or download this repository.
2. Download the Sentiment140 dataset CSV and move it into the `Data/` folder.
3. Install Python dependencies (recommended to use a virtual environment):

pip install pandas scikit-learn nltk matplotlib seaborn joblib wordcloud


4. Open and run the notebook `Notebooks/sentiment_analysis.ipynb` step by step.
5. The notebook performs data preprocessing, trains a Logistic Regression model, evaluates results, and saves the model in `Models/`.

---

## Key Results and Insights

- The model achieves about 75% accuracy for binary sentiment classification (positive vs negative).
- Top positive sentiment words identified include: `congrat`, `awesome`, `love`, `smile`, `thank`.
- Top negative sentiment words identified include: `sad`, `disappoint`, `upset`, `hurt`.
- Neutral sentiment examples are minimal or absent in this dataset.
- The notebook includes visualizations such as word clouds and confusion matrices to help understand predictions.

---

## Limitations and Future Work

- The current model is a **baseline classical approach**; accuracy can be improved by using more sophisticated models (SVM, Random Forest, or transformers like BERT).
- The dataset does not contain balanced neutral sentiments, limiting multi-class sentiment analysis.
- Additional features such as emoji analysis, hashtag extraction, or contextual understanding could enhance performance.

---

## Acknowledgements

- Sentiment140 dataset by [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140).
- Python packages used include scikit-learn, pandas, nltk, matplotlib, seaborn, joblib, and wordcloud.

---

## Contact

Created by Alby Anil  
Feel free to reach out for questions or collaborations!

---

*Thank you for checking out this project!*
