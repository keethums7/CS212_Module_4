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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
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
    word_pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000))
        ])

    # train the pipeline to the data
    word_pipe = word_pipe.fit(train_texts, train_labels)

    # run a prediction on the test text using the initial pipe (default - words - for TfidfVectorizer)
    word_prediction = word_pipe.predict(test_texts)

    # show the accuracy on the test data
    word_mean = np.mean(word_prediction == test_labels)
    print(f"\nSingle word features, accuracy: {100 * word_mean:.2f}%\n")

    # classification report & confusion matrix
    clf_report_tfidfv = metrics.classification_report(test_labels, word_prediction)
    clf_cf_matrix = metrics.confusion_matrix(test_labels, word_prediction)

    print(f"Single word features, Classification Report:\n{clf_report_tfidfv}")
    print(f"Default TfidfVectorizer, Confusion Matrix:\n{clf_cf_matrix}\n")

    # Step 2. Experiment 1: Which features work best?
     
    # We already have the pipeline built for the default TfidfVectorizer
    # just need to run the cross_val_score for it
    word_cvs = cross_val_score(word_pipe, train_texts, train_labels, cv=5)

    
    print(f"Single word features, cross_val_score (5) mean accuracy: {100 * word_cvs.mean():.2f}%") 
    print(f"Single word features, cross_val_score (5) std deviation: {100 * word_cvs.std():.2f}%\n")

    # now do the same thing for word and word pairs (ngram_range=(1, 2))
    # start by building the pipeline
    word_pair_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000))
        ])
    word_pair_pipe = word_pair_pipe.fit(train_texts, train_labels)
    word_pair_cvs = cross_val_score(word_pair_pipe, train_texts, train_labels, cv=5)

    print(f"Word pair features, cross_val_score (5) mean accuracy: {100 * word_pair_cvs.mean():.2f}%") 
    print(f"Word pair features, cross_val_score (5) std deviation: {100 * word_pair_cvs.std():.2f}%\n")

    # same things for characters (analyzer="char", ngram_range=(2, 4))
    char_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 4))),
        ("clf", LogisticRegression(max_iter=1000))
        ])
    char_pipe = char_pipe.fit(train_texts, train_labels)
    char_cvs = cross_val_score(char_pipe, train_texts, train_labels, cv=5)

    print(f"Character features, cross_val_score (5) mean accuracy: {100 * char_cvs.mean():.2f}%") 
    print(f"Character features, cross_val_score (5) std deviation: {100 * char_cvs.std():.2f}%\n")

    # and for words, this time with common english words removed (stop_words="english")
    stop_word_pipe = Pipeline([
                                  ("tfidf", TfidfVectorizer(stop_words="english")),
                                  ("clf", LogisticRegression(max_iter=1000))
                              ])
    stop_word_pipe = stop_word_pipe.fit(train_texts, train_labels)
    stop_word_cvs = cross_val_score(stop_word_pipe, train_texts, train_labels, cv=5)

    print(f"Single word features w/English stop words removed, cross_val_score (5) mean: \
          {100 * stop_word_cvs.mean():.2f}%\n") 
    print(f"Single word features w/English stop words removed, cross_val_score (5) std \
          deviation: {100 * stop_word_cvs.std():.2f}%\n")

    """
    Output:
    Single word features, cross_val_score (5) mean accuracy: 89.86%
    Single word features, cross_val_score (5) std deviation: 1.21%

    Word pair features, cross_val_score (5) mean accuracy: 89.94%
    Word pair features, cross_val_score (5) std deviation: 1.01%

    Character features, cross_val_score (5) mean accuracy: 92.39%
    Character features, cross_val_score (5) std deviation: 1.12%

    Single word features w/English stop words removed, cross_val_score (5) mean accuracy: 86.21%
    Single word features w/English stop words removed, cross_val_score (5) std deviation: 1.99%
    --------------------
    Questions:
    1. Which features were the best?
    - The feature set yielding the highest accuracy was the single characters (92.39%).
    - The feature set yielding the lowest accuracy was the single words with
        stop words removed (86.21%)
    - The difference between the two was 6.18% (92.39-86.21), more than the
        std deviation of either (1.12% | 1.99%)
    - While there wasn't a substantial difference between using single characters
        and words or word pairs as features, there is enough of a difference between
        single word features with english stop words removed and the other three
        methods to say that those three *appear* to be slightly better, arguably in
        their own tier, when compared to single words w/stop words removed.

    2. What happened when you removed the stop words? Does that surprise you?
    - It seems like removing stop words actually reduced the accuracy, which is surprising.
    - These are answers to questions, though, which means they're likely written in layman's
        terms and contain less dense, expert level vocabluary and a closer style to spoken
        English, which means removing the common stop words may yield less accurate results.
    """

    # Step 3. Look inside the classifier

    # print the top 15 strongest weighted "human" words and the same for "ChatGPT" words
    words = word_pipe.named_steps["tfidf"].get_feature_names_out()
    weights = word_pipe.named_steps["clf"].coef_[0] # coef_ holds the weights for LogisticRegression
    order = np.argsort(weights) # positions of the words, [0:15] range for human, [-15:-1] for ChatGPT

    print("\nTop 15 human-like words from the word-based pipeline:\n") # top 15 human words
    for i in list(words[order[:15]]):
        print(f"- {i}")

    print("\nTop 15 ChatGPT-like words from the word-based pipeline:\n") # top 15 ChatGPT words
    for i in list(words[order[-15:]]):
        print(f"- {i}")

    """
    Output:
    Top 15 human-like words from the word-based pipeline:

    - an
    - displaystyle
    - theory
    - from
    - by
    - with
    - only
    - 11
    - 10
    - sometimes
    - methods
    - engineering
    - general
    - usually
    - 12

    Top 15 ChatGPT-like words from the word-based pipeline:

    - tasks
    - important
    - is
    - involves
    - his
    - variety
    - they
    - the
    - can
    - including
    - used
    - that
    - it
    - to
    - and
    --------------------    
    Questions:
    1. Which words are about style?
    - The only word out of both lists that really seems to be about style looks like
        "displaystyle" on the human list.
    2. Which are artifacts of how the data was collected?
    - "displaystyle", "engineering", and especially "10", "11", and "12", all sound
        like possible artifacts since it's unlikely you'd find those words in just
        any general knowledge responses to questions whose answers are sourced
        from wiki pages
    """

    # Step 4. Tune the hyperparameters
    
    
    # GridSearchCV using the existing pipeline
    parameters = {
        "C": [100, 250, 500, 750, 1000],
        "vect__ngram_range": [(1, 1)], # default for tfidf
        "tfidf__use_idf": [True, True, True, True, True],
    }
    word_gs = GridSearchCV(word_pipe, parameters, cv=5, n_jobs=-1)

    # perform the grid search
    word_gs.fit(train_texts, train_labels)

    # report the best parameters
    print("Best parameters found:", word_gs.best_params_)
