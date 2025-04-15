import os
import argparse
import sys

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error


parser = argparse.ArgumentParser(description="Predict currency with linear regression.")

parser.add_argument('--data', '-d', type=str, required=True,
                    help="Specify an input dataset path.")

parser.add_argument('--observation_window', '-ow', type=int, default=3,
                    help="Specify observation window (day shift) in values.")

parser.add_argument('--predicted_currency', '-pc', type=str, required=True,
                    help="Specify currency to predict. Other currencies will be used for training.")

parser.add_argument('--plot', '-p', type=str, default='prediction.png',
                    help="Specify output plot path.")


def generate_train_data(df: pd.DataFrame, target: str, window: int) -> tuple[pd.DataFrame, pd.Series]:
    X_train: pd.DataFrame = df.copy()
    y_train = df[target].shift(-window).dropna()

    to_concat = [X_train]

    for i in range(1, window):
        df_shifted = X_train.shift(-i)
        df_shifted = df_shifted.rename(columns=lambda col: f'{col}_{i}')
        to_concat.append(df_shifted)

    X_train = pd.concat(to_concat, axis=1).iloc[y_train.index]

    return X_train, y_train


def generate_plots(X_train: pd.DataFrame, y_train: pd.Series, save_to: str, title: str):
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)

    fig, axs = plt.subplots(1, 2, figsize=(19.20, 10.80))
    fig.suptitle(title)

    for ax, X, y, name in zip(axs, (X_train, X_test), (y_train, y_test), ("Train", "Test")):
        ax.plot(y)
        ax.plot(y.index, pred := model.predict(X))
        ax.set_title(f"{name} MAPE: {mean_absolute_percentage_error(y, pred):.2f}")
        ax.set(xlabel="Day", ylabel="Price USD")

    fig.legend(["Actual", "Predicted"])
    fig.savefig(save_to)


def main():
    args = parser.parse_args()

    if not os.path.isfile(args.data):
        print(f'Dataset provided does not exist {args.data}')
        sys.exit()

    try:
        df: pd.DataFrame = pd.read_csv(args.data)
    except Exception:
        print(f'Failed to read {args.data} as csv.')
        sys.exit()

    if args.predicted_currency not in df.columns:
        print(f'There is no such column to predict as {args.predicted_currency}')
        sys.exit()

    X_train, y_train = generate_train_data(df, args.predicted_currency, args.observation_window)
    generate_plots(X_train, y_train, save_to=args.plot,
                   title=f"Linear Regression performance on crypto market ({args.predicted_currency.capitalize()})")


if __name__ == '__main__':
    main()
