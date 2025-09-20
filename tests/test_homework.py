"""Autograding script."""

import pkl

import pandas as pd  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore


def test_01():
    """Test the homework."""

    dataframe = pd.read_csv(
        "files/input/sentences.csv.zip",
        index_col=False,
        compression="zip",
    )

    with open("homework/clf.pkl", "rb") as file:
        clf = pkl.load(file)

    with open("homework/vectorizer.pkl", "rb") as file:
        vectorizer = pkl.load(file)

    accuracy = accuracy_score(
        y_true=dataframe.target,
        y_pred=clf.predict(vectorizer.transform(dataframe.phrase)),
    )

    assert accuracy > 0.854
