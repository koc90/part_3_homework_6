import os
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt


def read_data(filename):
    return pd.read_csv(os.path.join("data", filename), header=None)


def prepare_data(df: pd.DataFrame):

    columns = df.columns
    y = df[columns[0]].to_numpy()
    df = df.drop(columns=columns[0])

    X = df.to_numpy()

    return (X, y)


def sort_result(X, y):
    clusters = list(set(y))
    sorted_result = dict()

    for cluster in clusters:
        sorted_result[cluster] = []

    for result in zip(X, y):
        sorted_result[result[1]].append(list(result[0]))

    for key, value in sorted_result.items():
        sorted_result[key] = np.array(value)

    return sorted_result


def plot_result(sorted_result: dict[int | str : np.array]):
    legend = []
    for key, value in sorted_result.items():
        legend.append(str(key))
        plt.scatter(value[:, 0], value[:, 1])

    plt.legend(legend)


def main():
    files = ["data_2d.csv", "mnist.csv"]
    df = read_data(files[0])

    X, y = prepare_data(df)

    print(f"X = {X}")
    print(f"y = {y}")

    sorted_result = sort_result(X, y)

    print(sorted_result)
    plot_result(sorted_result)
    plt.show()


if __name__ == "__main__":
    main()
