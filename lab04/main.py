"""
Keith Wangler
CS212 - Fall 2026 - Brian Bird
Lab 4
10/25/2026

No AI/code completion was used whatsoever while updating this code from the
starter code to what you see currently.
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
    print(f"Single word features, Confusion Matrix:\n{clf_cf_matrix}\n")

    # Step 2. Experiment 1: Which features work best?
     
    # We already have the pipeline built for the default TfidfVectorizer
    # just need to run the cross_val_score for it
    word_cvs = cross_val_score(word_pipe, train_texts, train_labels, cv=5)

    
    print(f"Single word features, cross_val_score (5) mean accuracy:"\
        f"{100 * word_cvs.mean():.2f}%") 
    print(f"Single word features, cross_val_score (5) std deviation:"\
        f"{100 * word_cvs.std():.2f}%\n")

    # now do the same thing for word and word pairs (ngram_range=(1, 2))
    # start by building the pipeline
    word_pair_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000))
        ])
    word_pair_pipe = word_pair_pipe.fit(train_texts, train_labels)
    word_pair_cvs = cross_val_score(word_pair_pipe, train_texts, train_labels, cv=5)

    print(f"Word pair features, cross_val_score (5) mean accuracy:"\
        f"{100 * word_pair_cvs.mean():.2f}%")
    print(f"Word pair features, cross_val_score (5) std deviation:"\
        f"{100 * word_pair_cvs.std():.2f}%\n")

    # same things for characters (analyzer="char", ngram_range=(2, 4))
    char_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 4))),
        ("clf", LogisticRegression(max_iter=1000))
        ])
    char_pipe = char_pipe.fit(train_texts, train_labels)
    char_cvs = cross_val_score(char_pipe, train_texts, train_labels, cv=5)

    print(f"Character features, cross_val_score (5) mean accuracy:"\
          f"{100 * char_cvs.mean():.2f}%") 
    print(f"Character features, cross_val_score (5) std deviation:"\
          f"{100 * char_cvs.std():.2f}%\n")

    # and for words, this time with common english words removed (stop_words="english")
    stop_word_pipe = Pipeline([
                                  ("tfidf", TfidfVectorizer(stop_words="english")),
                                  ("clf", LogisticRegression(max_iter=1000))
                              ])
    stop_word_pipe = stop_word_pipe.fit(train_texts, train_labels)
    stop_word_cvs = cross_val_score(stop_word_pipe, train_texts, train_labels, cv=5)

    print(f"Single word features w/English stop words removed, cross_val_score (5) mean "\
        f"accuracy {100 * stop_word_cvs.mean():.2f}%") 
    print(f"Single word features w/English stop words removed, cross_val_score (5) std "\
        f"deviation: {100 * stop_word_cvs.std():.2f}%\n")

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
        ---------------
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
        ---------------
        Questions:
        1. Which words are about style?
        - The only word out of both lists that really seems to be about style looks like
            "displaystyle" on the human list.
        2. Which are artifacts of how the data was collected?
        - "displaystyle" and especially "10", "11", and "12", all sound like possible
            artifacts of the human side writing style since they are copied
            and pasted in from wikipedia articles and may include HTML/other artifacts
    """

    # Step 4. Tune the hyperparameters
    
    
    # GridSearchCV using the existing pipeline
    parameters = {
        "tfidf__ngram_range": [(1, 1), (1, 2)], # word, word-pair
        "tfidf__use_idf": (True, False), # with and without inverse document frequency
        "tfidf__min_df": [1, 2], # if an ngram appears only once or only twice ignore it
        "clf__C": (100, 250, 500, 750, 1000),
    }
    word_gs = GridSearchCV(estimator=word_pipe, param_grid=parameters, cv=5, n_jobs=-1)
    word_gs = word_gs.fit(train_texts, train_labels)

    # test its prediction on the test texts
    word_gs_predict = word_gs.predict(test_texts)
    word_gs_accuracy = np.mean(word_gs_predict == test_labels)

    # report the best parameters and cvs
    print("\nBest parameters found using GridSearchCV on training data: ")
    for i in word_gs.best_params_:
        print(f"- {i}: {word_gs.best_params_[i]}")

    print(f"\nCross-validation score: {100 * word_gs.best_score_:.2f}%")

    # report the accuracy using the best parameters
    print(f"\nAccuracy on test data while using the best parameters: {100 * word_gs_accuracy:.2f}%")

    """
        Output:
        Best parameters found using GridSearchCV on training data: 
        - clf__C: 500
        - tfidf__min_df: 2
        - tfidf__ngram_range: (1, 2)
        - tfidf__use_idf: True

        Cross-validation score: 94.61%

        Accuracy on test data while using the best parameters: 94.79%
    """

    # Step 5. Experiment 2: A new topic
    
    # load questions from open_qa.jsonl and load into new test data
    new_questions, new_human_answers, new_chatgpt_answers = load_hc3("data/open_qa.jsonl")
    new_questions_numbers = list(range(len(new_questions)))
    new_test_texts, new_test_labels = make_samples(new_human_answers, new_chatgpt_answers, new_questions_numbers)

    # test the trained model above on the new test data,
    # grab the accuracy, classification report, and confusion matrix
    new_word_gs_predict = word_gs.predict(new_test_texts)
    new_word_gs_accuracy = np.mean(new_word_gs_predict == new_test_labels)
    new_word_gs_cr = metrics.classification_report(new_test_labels, new_word_gs_predict)
    new_word_gs_cm = metrics.confusion_matrix(new_test_labels, new_word_gs_predict)

    # print out all of the above
    print(
      f"Loaded data from 'data/open_qa.jsonl'..."
      f"testing model trained using best params:\n"\
      f"Accuracy: {100 * new_word_gs_accuracy:.2f}%\n"\
      f"Classification report:\n{new_word_gs_cr}\n"\
      f"Confusion Matrix:\n{new_word_gs_cm}"
      )
  
    """
        Output:
        Loaded data from 'data/open_qa.jsonl'... testing model trained using best params:
        Accuracy: 78.48%
        Classification report:
                      precision    recall  f1-score   support

                   0       0.71      0.97      0.82      1187
                   1       0.95      0.60      0.74      1187

            accuracy                           0.78      2374
           macro avg       0.83      0.78      0.78      2374
        weighted avg       0.83      0.78      0.78      2374

        Confusion Matrix:
        [[1150   37]
         [ 474  713]]
        ---------------
        Questions:
        1. How much did the accuracy drop compared with the wiki_csai test set?
        - Accuracy dropped from 94.79% to 78.48%, a difference of 16.31%
        2. Which of the two classes did the model get wrong more often?
            Compare the lengths of the human and ChatGPT answers in the two
            files (the median number of words is a good measure) and explain
            what you see
        - It looks like it predicted human text more accurately if I'm reading it
        correctly (0=human, 1=chatgpt)
        - The ChatGPT answers are consistently much longer than the human answers,
            which makes some sense in terms of the effort needed to write longer,
            more in-depth and verbose answers.
    """

    # Step 6. Error analysis
    
    # print five (5) answers from the wiki_csai test set that the tuned model got wrong

    # loop through the prediction object until 
    # we know we have 5 labels that were incorrect
    # wrong_count = 0
       # i = 0
    # while wrong_count < 5:
    #     for

    # iterate through the test labels, check if the index
    # doesn't match with the predicted value 
    # (0 for human, 1 for ChatGPT)
    # until 5 wrong answers are accumulated
    wrong_answers = []
    i = 0 # starting index for the loop

    while len(wrong_answers) < 5:
        if word_prediction[i] != test_labels[i]:
            answer = {
                'text': test_texts[i],
                'predicted_label': word_prediction[i],
                'true_label': test_labels[i]
            }
            wrong_answers.append(answer)
        i += 1 # increment the index and check again

    print("5 wrong answers:\n", wrong_answers)

    """
    Output:
    [
    {'text': 'Robert Ettinger was a science fiction author and cryonics advocate who is best known for his books 
    "The Prospect of Immortality"and "Man into Superman." He was born in 1918 and grew up in New Jersey. 
    He served in World War II and later earned a bachelor\'s degree in physics from the University of Michigan.
    \n\n\n\nEttinger became interested in the idea of cryonics, or the freezing of human bodies 
    in the hope of reviving them in the future when medical technology has advanced, 
    after reading about it in a science fiction magazine in the 1950s. 
    He wrote "The Prospect of Immortality" in 1964, which became a seminaltext in the cryonics movement. 
    In the book, he argued that cryonics offered the possibility of extending human life indefinitely 
    and that it was a viable option for people who wanted to preserve their bodies in the hope of 
    being revived in the future.\n\n\n\nEttinger founded theCryonics Institute in 1976, 
    a non-profit organization that offers cryopreservation services to its members. 
    He died in 2011 at the age of 92 and was cryopreserved at the institute. 
    His work and ideas about cryonics have been both praised and criticized by scientists and philosophers.', 
    'predicted_label': np.int64(0), 'true_label': 1},
    {'text': "Animage sensor is a device that converts an optical image into an electronic signal. 
    It is used in digital cameras, scanners, and other image-capturing devices to capture and record images.
    \n\n\n\nAn image sensor consists of a matrix of photodiodes, which are light-sensitive elements 
    that convert light into an electrical current. When light enters the camera and hits the image sensor, 
    it is absorbed by the photodiodes, which generate an electrical current proportional to the intensity of the light. 
    The image sensor converts this electrical current into digital data, which is then processed 
    by the camera's image processing unit to produce a digital image.
    \n\n\n\nThere are two main types of image sensors: CCD (charge-coupled device) 
    and CMOS (complementary metal-oxide-semiconductor). CCD sensors are generally more expensive to manufacture, 
    but they produce higher quality images with less noise. CMOS sensors are 
    less expensive to manufacture and consume less power, but they may produce images with more noise.
    \n\n\n\nIn summary, an image sensor is a crucial component of any image-capturing device, 
    as it converts the light that enters the device into a digital representation ofthe scene that can be stored and manipulated.", 
    'predicted_label': np.int64(0), 'true_label': 1},
    {'text': 'In mathematics, a polynomial isan expression consisting of indeterminates (also called variables) 
    and coefficients, that involves only the operations of addition, subtraction, multiplication, 
    and positive-integer powers of variables. An example of a polynomial of a single indeterminate x is x2 − 4x + 7. 
    An example with three indeterminates is x3 + 2xyz2 − yz + 1.
    \nPolynomials appear in many areas of mathematics and science. For example, 
    they are used to form polynomial equations, which encode a wide range of problems, 
    from elementary word problems to complicated scientific problems; they are used to define polynomial functions, 
    which appear in settingsranging from basic chemistry and physics to economics and social science; 
    they are used in calculus and numerical analysis to approximate other functions. In advanced mathematics, 
    polynomials are used to construct polynomial rings and algebraic varieties, which are central concepts 
    in algebra and algebraic geometry.', 
    'predicted_label': np.int64(1), 'true_label': 0},
    {'text': 'In mathematics, a polynomial is an expression consisting of variables (also called indeterminates) 
    and coefficients, that involves only the operations of addition, subtraction, and multiplication, 
    and sometimes division (though this is not always allowed). For example, the polynomial 3x^2 + 2x - 5 
    is a polynomial in the variable x. The highest power of the variable in a polynomial is called 
    the degree of the polynomial. In this case, the degree of the polynomial is 2 because the highest power of x is 2.
    \n\n\n\nA multivariate polynomial is a polynomial with more than one variable. 
    For example,the expression 2x^2 + y^3 - 3z + 4 is a multivariate polynomial in the variables x, y, and z. 
    The degree of a multivariate polynomial is the highest power of the variables in the polynomial. 
    In this case, the degree of the polynomial is 3 because the highest power of any of the variables is 3 (in the term y^3).
    \n\n\n\nMultivariate polynomials can be useful for modeling relationships between multiple variables. 
    For example, a polynomial with two variables, such as 3x^2 + 2xy + y^2, 
    could be used to model the relationship between two variables, such as the price and demand for a product.', 
    'predicted_label': np.int64(0), 'true_label': 1},
    {'text': 'Social intelligence is the capacity to know oneself and to know others. 
    Social intelligence is learned and develops from experience with people and learning from success and failures in social settings.
    Social intelligence is the ability to understand your own and others actions. 
    It is also known as "tact" or "common sense", or "street smarts".
    [citation needed] It is an important interpersonal skill that will help individuals succeed in all aspects of their lives.', 
    'predicted_label': np.int64(1), 'true_label': 0}]

    Questions:
    1. What do the mistakes have in common?
        - For the answers that had predicted values of 0 (human) and true values of chatGPT (1),
        it's mostly clear that there are triplicate new line characters throughout those answers.
        - For the others (predicted 1 (chatGPT) true value 0 (human)), I don't know if I really
        see a discernable pattern or common thread.
    2. Are any of them hard even for you to judge?
        - Yes, as stated in question 1 - the answers that were predicted to be chatGPT but were
        actually human...I'm not exactly sure what it was seeing, perhaps it's the mathematical
        terms or the "[citation needed]" that would complicate the usual pattern that the 
        prediction relies on for its basis/logic.
    """