import pandas as pd


def main():
    data = pd.read_csv('Reviews.csv')

    print("Product count: ", data['ProductId'].nunique())
    print("User count: ", data['UserId'].nunique())


if __name__ == "__main__":
    main()
