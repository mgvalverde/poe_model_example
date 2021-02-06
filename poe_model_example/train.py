import pandas as pd
from pathlib import Path
import numpy as np
import click
import pickle
from datetime import datetime
from sklearn.datasets import load_wine
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('train')

model_folder = Path().home() / "models"
data_folder = Path().home() / "data"


def eval_metrics(actual, pred):
    prf = precision_recall_fscore_support(actual, pred, average="weighted")[:-1]
    return {k: round(v, 4) for k, v in zip(("prec", "recall", "f1"), prf)}


def model_tracking_report(params, metrics, model_id, folder):
    cols = pd.MultiIndex.from_tuples(
        [*[("params", k) for k in params.keys()], *[("metrics", k) for k in metrics.keys()]])
    df_metrics = pd.DataFrame([*params.values(), *metrics.values()], index=cols,
                              columns=pd.Index([model_id], name="model_id")).T
    fpath = folder / "metrics.csv"
    if fpath.exists():
        df_metrics.to_csv(folder / "metrics.csv", mode="a", header=None)
    else:
        df_metrics.to_csv(folder / "metrics.csv", mode="a")
    return df_metrics



@click.command()
@click.option('-n', '--n_estimators', default=100, help='Number of RF estimators')
def train(n_estimators: int):
    """
    Train RF model to predict wine quality
    """
    model_id = datetime.now().strftime("%Y%M%d.%H%M%S")
    model_folder.mkdir(parents=True, exist_ok=True)

    # Data load
    X_train = pd.read_csv(data_folder / "X_train.csv")
    X_test = pd.read_csv(data_folder / "X_test.csv")
    y_train = pd.read_csv(data_folder / "y_train.csv")
    y_test = pd.read_csv(data_folder / "y_test.csv")
    logger.info("Data Loaded!")

    # Train
    fc = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    fc = fc.fit(X_train, y_train.values.ravel())
    logger.info("Model trained!")

    # Metrics evaluation
    y_pred = fc.predict(X_test)

    metrics = eval_metrics(y_test, y_pred)
    df_metrics = model_tracking_report({"n_estimators":n_estimators}, metrics, model_id, model_folder)
    logger.info("Metrics saved!")

    # Model persinstance
    with open(str(model_folder / f'rf_{model_id}.pkl'), 'wb') as f:
        pickle.dump(fc, f)
    logger.info(f"Model {model_id} saved!")

if __name__ == "__main__":
    train()
