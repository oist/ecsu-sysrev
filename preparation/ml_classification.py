import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier


def tutorial():
    from sklearn.datasets import fetch_20newsgroups
    twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
    twenty_train.target_names # list of str of categories
    print("\n".join(twenty_train.data[0].split("\n")[:3])) #prints first line of the first data file
    # twenty_train.data -> list of str

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])
    text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

    import numpy as np
    twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
    predicted = text_clf.predict(twenty_test.data)
    print(np.mean(predicted == twenty_test.target))

def classify_aggregationA_NB():
    import json
    papers_json = json.load(open('data/aggregated_A_annotated.json'))    
    data = [' '.join([x['Title'],x.get('Abstract','')]) for x in papers_json]
    target = [x['Accepted'] for x in papers_json]

    test_size = 0.20
    runs = 100
    runs_accuracy = []

    pipe = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ]) # 0.87

    # pipe = Pipeline([
    #     ('vect', CountVectorizer(ngram_range=(1,3))),
    #     ('tfidf', TfidfTransformer()),
    #     ('clf', MultinomialNB()),
    # ]) # 0.87

    # pipe = Pipeline([
    #     ('vect', CountVectorizer(stop_words='english')),
    #     ('tfidf', TfidfTransformer()),
    #     ('clf', MultinomialNB()),
    # ]) # 0.88

    # from stemmed_count_vect import stemmed_count_vect
    # pipe = Pipeline([
    #     ('vect', stemmed_count_vect),
    #     ('tfidf', TfidfTransformer()),
    #     ('clf', MultinomialNB()),
    # ]) # 0.88


    for _ in range(runs):

        data_train, data_test, target_train, target_test = train_test_split(
            data, target, test_size=test_size) #random_state=42

        text_clf = pipe.fit(data_train, target_train)

        predicted = text_clf.predict(data_test)
        runs_accuracy.append(np.mean(predicted == target_test))

    print(np.mean(runs_accuracy))


def classify_aggregationA_SVM():
    import json

    def plot_coefficients(classifier, feature_names, top_features=30):
        import matplotlib.pyplot as plt
        coef = classifier.coef_.ravel()
        top_positive_coefficients = np.argsort(coef)[-top_features:]
        top_negative_coefficients = np.argsort(coef)[:top_features]
        top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
        # create plot
        plt.figure(figsize=(15, 5))
        colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
        plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
        feature_names = np.array(feature_names)
        plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
        plt.show()
    
    papers_json = json.load(open('data/aggregated_A_annotated.json'))    
    data = [' '.join([x['Title'],x.get('Abstract','')]) for x in papers_json]
    target = [x['Accepted'] for x in papers_json]
    print('Total papers, accepted : {} {}'.format(len(target), sum([1 for x in target if x])))    

    test_size = 0.20
    runs = 100
    runs_accuracy = []

    # cv = CountVectorizer()
    cv = CountVectorizer(stop_words='english')
    pipe = Pipeline([
        ('vect', cv),
        ('tfidf', TfidfTransformer()),
        ('clf-svm', SGDClassifier(
            loss='hinge', penalty='l2', alpha=1e-3, n_iter_no_change=5, random_state=42)),
    ]) # 0.92

    # pipe = Pipeline([
    #     ('vect', CountVectorizer()),
    #     ('tfidf', TfidfTransformer()),
    #     ('clf-svm', SGDClassifier(
    #         loss='hinge', penalty='l2', alpha=1e-3, n_iter_no_change=5, random_state=42)),
    # ]) # 0.90

    for _ in range(runs):

        data_train, data_test, target_train, target_test = train_test_split(
            data, target, test_size=test_size) #random_state=42

        _ = pipe.fit(data_train, target_train)

        predicted = pipe.predict(data_test)
        runs_accuracy.append(np.mean(predicted == target_test))

    model = pipe.named_steps['clf-svm']
    plot_coefficients(model, cv.get_feature_names())

    print(np.mean(runs_accuracy))


if __name__ == '__main__':
    # tutorial()
    # classify_aggregationA_NB()
    classify_aggregationA_SVM()
    