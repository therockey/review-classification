from collections import Counter
import nltk
import pandas as pd
import pylab as pl
from matplotlib import pyplot as plt
import seaborn as sns


def main():
    data = pd.read_csv('Reviews_cleaned.csv')

    print("Product count: ", data['ProductId'].nunique())
    print("User count: ", data['UserId'].nunique())

    # Calculate the length of each review
    data['review_length'] = data['Text'].apply(review_length)
    data['label'] = data['Score'].apply(lambda x: 'positive' if x > 3 else 'negative' if x < 3 else 'neutral')
    review_counts = data['label'].value_counts()
    for label, count in review_counts.items():
        print(f"{label}: {count} ({count / len(data) * 100:.2f}%)")

    # Plot the distribution of review lengths
    plt.figure(figsize=(10, 6))
    sns.distplot(data['review_length'], kde=False, color='blue', hist_kws={"rwidth": 0.8})
    plt.title('Dystrybucja długości recenzji')
    plt.xlabel('Długość recenzji w słowach')
    plt.ylabel('Częstotliwość')
    plt.yscale('log')
    pl.xlim([1, 1000])
    plt.show()

    words = []

    nltk.download('punkt')

    for review in data['Text']:
        if not isinstance(review, str):
            continue

        words.extend(nltk.word_tokenize(review))

    word_freq = Counter(words)

    most_common_words = word_freq.most_common(10)

    words, frequencies = zip(*most_common_words)

    # Wykres słupkowy najczęściej występujących słów
    plt.figure(figsize=(10, 6))
    plt.bar(words, frequencies, color='blue')
    plt.title('Najczęściej występujące słowa w recenzjach')
    plt.xlabel('Słowa')
    plt.ylabel('Liczba wystąpień')
    plt.show()

    # Macierz korelacji długości recenzji i ocen
    plt.figure(figsize=(10, 6))
    sns.heatmap(data[['review_length', 'Score']].corr(), annot=True, cmap='coolwarm')
    plt.title('Macierz korelacji długości recenzji i ocen')
    plt.show()


def review_length(review):
    if not isinstance(review, str):
        return 0

    return len(review.split())


if __name__ == "__main__":
    main()
