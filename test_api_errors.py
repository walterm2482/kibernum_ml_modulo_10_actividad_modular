"""
test_api_errors.py
------------------
Pruebas negativas para la API Flask.

Estas pruebas verifican el manejo correcto de errores en los endpoints:
- ``POST /predict`` con clave incorrecta.
- ``POST /predict`` con número inválido de características.

Ejecutar con:
    pytest -v test_api_errors.py
"""

import json
from app import create_app

# --- Configuración del cliente de prueba -------------------------------------
app = create_app()
client = app.test_client()


def test_bad_key():
    """
    Prueba ``POST /predict`` con una clave de JSON incorrecta.

    Envía un payload que no contiene las claves esperadas
    ('features' o 'instances') y valida que el servidor responda con error 400.

    Returns
    -------
    None
    """
    response = client.post(
        "/predict",
        data=json.dumps({"x": [1, 2, 3]}),
        content_type="application/json",
    )
    assert response.status_code == 400


def test_bad_shape():
    """
    Prueba ``POST /predict`` con una cantidad incorrecta de características.

    Envía una lista demasiado corta para el modelo y valida que el
    servidor retorne un error de validación (400).

    Returns
    -------
    None
    """
    response = client.post(
        "/predict",
        data=json.dumps({"features": [1, 2]}),
        content_type="application/json",
    )
    assert response.status_code == 400
