import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import dump, load
import os
from imblearn.over_sampling import SMOTE

TRAIN_SIZE = 0.8
RANDOM_STATE = 42
DATA_FILE = 'Reviews_cleaned.csv'
WEIGHTED = False

ALGORITHMS = [{'name': 'Naive-Bayes',
               'model': MultinomialNB,
               'args': {},
               'trained_model': None,
               'result': []
               },
              {'name': 'C4-5',
               'model': DecisionTreeClassifier,
               'args': {'criterion': 'entropy', 'random_state': 42},
               'result': []
               },
              ]


def main():
    # 1. Pobranie danych
    data = pd.read_csv(DATA_FILE)

    print(f"Successfully read file {DATA_FILE}...")

    # 2. Wstępne przetwarzanie
    data = data.dropna(subset=['Text', 'Score'])
    data['label'] = data['Score'].apply(lambda x: 'positive' if x > 3 else 'negative' if x < 3 else 'neutral')

    print(f"Successfully dropped NaNs and created labels...")

    # 3. Tokenizacja i TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    x = tfidf.fit_transform(data['Text'])
    y = data['label']

    print(f"Successfully tokenized and calculated TF-IDF...")

    # 4. Balansowanie danych
    smote = SMOTE()
    x_res, y_res = smote.fit_resample(x, y)

    print(f"Successfully balanced data...")

    # 5. Podział danych na zestawy treningowe i testowe
    x_train, x_test, y_train, y_test, weights_train, weights_test = train_test_split(x_res, y_res,
                                                                                     test_size=1 - TRAIN_SIZE,
                                                                                     random_state=RANDOM_STATE)

    print(f"Successfully split data into training and testing sets...")

    # 6. Trenowanie modelów
    for algorithm in ALGORITHMS:
        if WEIGHTED:
            algorithm['trained_model'] = train_model(algorithm, x_train, y_train, weights_train)
        else:
            algorithm['trained_model'] = train_model(algorithm, x_train, y_train)

    # 7. Predykcja
    for algorithm in ALGORITHMS:
        algorithm['result'] = algorithm['trained_model'].predict(x_test)

    # 8. Ewaluacja
    for algorithm in ALGORITHMS:
        print(algorithm['name'])
        print(classification_report(y_test, algorithm['result']))
        print_confusion_matrix(y_test, algorithm['result'], algorithm['name'], WEIGHTED)


def print_confusion_matrix(y_test, y_pred, algorithm_name, weighted=False):
    conf_matrix = confusion_matrix(y_test, y_pred, labels=['positive', 'neutral', 'negative'])
    sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=['positive', 'neutral', 'negative'],
                yticklabels=['positive', 'neutral', 'negative'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion matrix for {algorithm_name}, weighted={weighted}')
    plt.show()


def train_model(algorithm, x_train, y_train, weights=None):
    if not os.path.exists(f'{algorithm["name"]}_{len(y_train)}_{"weighted" if weights else "unweighted"}.joblib'):
        print(
            f'Training {"weighted" if weights else "unweighted"} model {algorithm["name"]} on sample size {len(y_train)}...')
        model = algorithm['model'](**algorithm['args'])
        model.fit(x_train, y_train, sample_weight=weights)

        dump(model, f'{algorithm["name"]}_{len(y_train)}_{"weighted" if weights else "unweighted"}.joblib')
        return model

    print(
        f'Model file found, loading {"weighted" if weights else "unweighted"} model {algorithm["name"]} trained on sample size {len(y_train)}...')

    return load(f'{algorithm["name"]}_{len(y_train)}_{"weighted" if weights else "unweighted"}.joblib')


if __name__ == "__main__":
    main()
