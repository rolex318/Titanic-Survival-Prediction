import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def load_titanic_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].copy()

    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].to_numpy()
    y = df['Survived'].to_numpy()

    return X, y


def train_and_save_model():
    dataset_path = Path('dataset/titanic.csv')
    model_dir = Path('models')
    model_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_titanic_dataset(str(dataset_path))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42)
    model.fit(X_scaled, y)

    with open(model_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    with open(model_dir / 'model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print('Saved scaler to models/scaler.pkl')
    print('Saved model to models/model.pkl')
    print('Current scikit-learn version:', sklearn.__version__)


if __name__ == '__main__':
    train_and_save_model()
