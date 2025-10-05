"""
train_model.py
---------------
Entrena y serializa un modelo de clasificación usando el dataset Breast Cancer de sklearn.
El modelo resultante se guarda como 'model.joblib' para ser utilizado en una API Flask.
"""

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from logger import get_logger

# --- Logger global ------------------------------------------------------------
logger = get_logger("mlops.train")


def load_data():
    """
    Carga el dataset Breast Cancer y lo divide en conjuntos de entrenamiento y prueba.

    Returns
    -------
    X_train : ndarray
        Características de entrenamiento.
    X_test : ndarray
        Características de prueba.
    y_train : ndarray
        Etiquetas de entrenamiento.
    y_test : ndarray
        Etiquetas de prueba.
    """
    logger.info("Cargando dataset Breast Cancer...")
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    logger.info(
        "Datos cargados: %d muestras de entrenamiento, %d de prueba",
        X_train.shape[0],
        X_test.shape[0],
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Entrena un modelo Random Forest con los datos de entrenamiento.

    Parameters
    ----------
    X_train : ndarray
        Características de entrenamiento.
    y_train : ndarray
        Etiquetas de entrenamiento.

    Returns
    -------
    model : RandomForestClassifier
        Modelo entrenado.
    """
    logger.info("Entrenando modelo RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    logger.info("Entrenamiento completado.")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo en el conjunto de prueba y muestra su precisión.

    Parameters
    ----------
    model : RandomForestClassifier
        Modelo entrenado.
    X_test : ndarray
        Características de prueba.
    y_test : ndarray
        Etiquetas de prueba.

    Returns
    -------
    acc : float
        Precisión del modelo.
    """
    logger.info("Evaluando modelo...")
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    logger.info("Precisión obtenida: %.2f", acc)
    return acc


def save_model(model, filename="model.joblib"):
    """
    Guarda el modelo entrenado en disco.

    Parameters
    ----------
    model : RandomForestClassifier
        Modelo entrenado.
    filename : str, optional
        Nombre del archivo donde guardar el modelo (por defecto 'model.joblib').
    """
    joblib.dump(model, filename)
    logger.info("Modelo guardado en '%s'", filename)


def main():
    """
    Ejecuta el flujo principal:
    carga datos, entrena el modelo, evalúa el desempeño y guarda el resultado.
    """
    try:
        X_train, X_test, y_train, y_test = load_data()
        model = train_model(X_train, y_train)
        acc = evaluate_model(model, X_test, y_test)
        save_model(model)
        logger.info("Proceso completado con éxito. Accuracy final: %.2f", acc)
    except Exception as e:
        logger.exception("Error durante el entrenamiento: %s", e)


if __name__ == "__main__":
    main()
