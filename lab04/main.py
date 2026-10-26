"""
Keith Wangler
CS212 - Fall 2026 - Brian Bird
Lab 4
10/25/2026
"""

import json
import numpy as np

from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def load_hc3(path):
    """Read an HC3 file. Returns three lists with one item per question:
    the questions, the first human answer and the first ChatGPT answer."""
    questions = []
    human_answers = []
    chatgpt_answers = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            questions.append(item["question"])
            human_answers.append(item["human_answers"][0])
            chatgpt_answers.append(item["chatgpt_answers"][0])
    return questions, human_answers, chatgpt_answers

def make_samples(human_answers, chatgpt_answers, question_numbers):
    """Build the texts and labels (0 = human, 1 = ChatGPT) for the chosen questions."""
    texts = []
    labels = []
    for number in question_numbers:
        texts.append(human_answers[number])
        labels.append(0)
        texts.append(chatgpt_answers[number])
        labels.append(1)
    return texts, labels

if __name__ == "__main__":
    # load questions 
    questions, human_answers, chatgpt_answers = load_hc3("data/wiki_csai.jsonl")
    print(len(questions), "questions")

    # Split by question number, so both answers to a question stay together.
    question_numbers = list(range(len(questions)))
    train_numbers, test_numbers = train_test_split(
        question_numbers, test_size=0.25, random_state=42)

    train_texts, train_labels = make_samples(human_answers, chatgpt_answers, train_numbers)
    test_texts, test_labels = make_samples(human_answers, chatgpt_answers, test_numbers)
    print(len(train_texts), "training answers,", len(test_texts), "test answers")
    print("A human answer:  ", human_answers[0][:70])
    print("A ChatGPT answer:", chatgpt_answers[0][:70])

    # Step 1. Baseline

    # build pipeline to vectorize, transform, & classify
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000))
        ])

    # train the pipeline to the data
    pipe.fit(train_texts, train_labels)

    # run a prediction on the test text using the initial pipe (default - words - for tfidf)
    tfidf_prediction = pipe.predict(test_texts)

    # show the accuracy on the test data
    words_mean = np.mean(tfidf_prediction == test_labels)
    print(f"\nDefault TfidfVectorizer, accuracy: {100 * words_mean:.2f}%\n")

    # classification report & confusion matrix
    clf_report_tfidf = metrics.classification_report(test_labels, tfidf_prediction)
    clf_cf_matrix = metrics.confusion_matrix(test_labels, tfidf_prediction)

    print("Default TfidfVectorizer, Classification Report:\n", clf_report_tfidf)
    print("Default TfidfVectorizer, Confusion Matrix:\n", clf_cf_matrix)

    # Step 2. Experiment 1: Which features work best?