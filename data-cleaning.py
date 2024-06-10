import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup

DATA_FILE = 'Reviews.csv'
OUTPUT_FILE = 'Reviews_cleaned.csv'


def main():
    # 1. Pobieranie stop listy słów
    nltk.download('stopwords')
    nltk.download('wordnet')

    # 2. Inicjalizacja lematyzatora
    lemmatizer = WordNetLemmatizer()

    # 3. Ustawienie listy stop słów na angielskie
    stop_words = set(stopwords.words('english'))

    # 4. Wczytanie danych z pliku CSV do DataFrame
    df = pd.read_csv(DATA_FILE)

    # 5. Usunięcie wierszy z brakującymi danymi w kolumnach 'review' i 'score'
    df = df.dropna(subset=['Text', 'Score'])

    # 5. Stworzenie listy recenzji z kolumny 'Text'
    reviews = df['Text'].tolist()

    # 6. Aplikowanie funkcji czyszczącej recenzje do każdej recenzji
    cleaned_reviews = [clean_review(review, lemmatizer, stop_words) for review in reviews]

    # 7. Zamiana kolumny 'Text' na 'cleaned_review'
    df['Text'] = cleaned_reviews

    # 8. Zapisanie DataFrame do pliku CSV
    df.to_csv(OUTPUT_FILE, index=False)


def clean_review(review, lemmatizer, stop_words):
    # 6.1. Usunięcie tagów HTML
    review = BeautifulSoup(review, "html.parser").get_text()

    # 6.2. Usunięcie znaków specjalnych
    review = re.sub("[^a-zA-Z]", " ", review)

    # 6.3. Zamiana na małe litery
    review = review.lower()

    # 6.4. Rozbicie recenzji na słowa
    words = review.split()

    # 6.5. Lematyzacja i usunięcie stop słów
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    # 6.6. Połączenie słów w całość
    cleaned_review = " ".join(words)

    return cleaned_review


if __name__ == '__main__':
    main()
