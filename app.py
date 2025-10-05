"""
app.py
------
API REST con Flask para exponer un modelo sklearn serializado.
Rutas:
- GET  /          : estado del servicio
- POST /predict   : predicción para una o varias instancias

El modelo se carga desde MODEL_PATH (env) o 'model.joblib' por defecto.
Entrada aceptada:
- {"features": [..]}  -> una instancia
- {"instances": [[..], [..]]} -> varias instancias
"""

import os
from typing import List, Optional, Any, Dict

import joblib
import numpy as np
from flask import Flask, request, jsonify

from logger import get_logger   # logger centralizado

# --- Logger global ------------------------------------------------------------
logger = get_logger("mlops.api")


# --- Utilidades ---------------------------------------------------------------
def _as_2d(array_like: List[List[float]]) -> np.ndarray:
    """
    Convierte una lista a arreglo 2D de float.

    Parameters
    ----------
    array_like : list of list of float
        Datos de entrada.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Arreglo 2D listo para el modelo.

    Raises
    ------
    ValueError
        Si la entrada no puede convertirse a 2D float.
    """
    X = np.asarray(array_like, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError("La entrada debe ser 1D o 2D.")
    return X


def _validate_shape(X: np.ndarray, expected_n_features: int) -> None:
    """
    Valida la dimensionalidad contra n_features del modelo.

    Parameters
    ----------
    X : ndarray
        Datos de entrada.
    expected_n_features : int
        Número de características esperadas por el modelo.

    Raises
    ------
    ValueError
        Si el número de características no coincide.
    """
    if X.shape[1] != expected_n_features:
        raise ValueError(
            f"Se esperaban {expected_n_features} características, "
            f"pero se recibieron {X.shape[1]}."
        )


# --- Carga de modelo ----------------------------------------------------------
def load_model(model_path: Optional[str] = None):
    """
    Carga el modelo serializado.

    Parameters
    ----------
    model_path : str, optional
        Ruta al archivo del modelo. Si no se entrega, se usa la variable
        de entorno MODEL_PATH o 'model.joblib'.

    Returns
    -------
    model : sklearn.base.BaseEstimator
        Modelo cargado.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe.
    """
    path = model_path or os.getenv("MODEL_PATH", "model.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el modelo en: {path}")
    model = joblib.load(path)
    logger.info("Modelo cargado desde %s", path)
    return model


# --- Fábrica de la aplicación -------------------------------------------------
def create_app(model=None) -> Flask:
    """
    Crea la aplicación Flask y registra rutas y manejadores. Compatible con Flask 3.x
    (sin before_first_request).

    Parameters
    ----------
    model : sklearn.base.BaseEstimator, optional
        Modelo previamente cargado. Si es None, se carga desde disco al crear la app.

    Returns
    -------
    app : Flask
        Aplicación Flask lista para ejecutar.
    """
    app = Flask(__name__)

    # Carga inmediata del modelo
    _model = {"obj": model or load_model()}

    # Mapeo de clases del dataset Breast Cancer: 0=malignant, 1=benign
    CLASS_MAP = {0: "malignant", 1: "benign"}

    @app.get("/")
    def health() -> Any:
        """Devuelve el estado del servicio."""
        info: Dict[str, Any] = {
            "status": "ok",
            "model_loaded": _model["obj"] is not None,
            "model_path": os.getenv("MODEL_PATH", "model.joblib"),
            "version": "1.0.0",
        }
        return jsonify(info), 200

    @app.post("/predict")
    def predict() -> Any:
        """Realiza predicciones para una o varias instancias."""
        try:
            payload = request.get_json(silent=True)
            if payload is None:
                return jsonify(error="JSON inválido o ausente."), 400

            if "features" in payload:
                X = _as_2d(payload["features"])
            elif "instances" in payload:
                X = _as_2d(payload["instances"])
            else:
                return jsonify(error="Proporcione 'features' o 'instances'."), 400

            mdl = _model["obj"]
            _validate_shape(X, getattr(mdl, "n_features_in_", X.shape[1]))

            y_pred = mdl.predict(X).tolist()
            response: Dict[str, Any] = {
                "predictions": y_pred,
                "classes": [CLASS_MAP.get(int(c), int(c)) for c in y_pred],
            }

            if hasattr(mdl, "predict_proba"):
                response["probabilities"] = mdl.predict_proba(X).tolist()

            return jsonify(response), 200

        except ValueError as ve:
            logger.warning("Error de validación: %s", ve)
            return jsonify(error=str(ve)), 400
        except FileNotFoundError as fe:
            logger.error("Modelo no disponible: %s", fe)
            return jsonify(error=str(fe)), 500
        except Exception:
            logger.exception("Fallo inesperado")
            return jsonify(error="Error interno del servidor."), 500

    @app.errorhandler(404)
    def not_found(_e):
        return jsonify(error="Ruta no encontrada."), 404

    @app.errorhandler(405)
    def method_not_allowed(_e):
        return jsonify(error="Método no permitido."), 405

    return app


# --- Punto de entrada ---------------------------------------------------------
if __name__ == "__main__":
    """
    Ejecuta el servidor de desarrollo.
    Para producción use un WSGI como gunicorn:
        gunicorn -w 2 -b 0.0.0.0:8000 app:create_app()
    """
    app = create_app()
    logger.info("Iniciando servidor Flask en puerto %s", os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)
