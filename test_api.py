"""
test_api.py
-----------
Pruebas unitarias para la API Flask que expone el modelo de predicción.

Estas pruebas verifican el funcionamiento básico de los endpoints:
- ``GET /``: comprueba el estado del servicio y la carga del modelo.
- ``POST /predict``: valida que se realicen predicciones correctamente
  a partir de un JSON con características numéricas.

Ejecutar con:
    pytest -v test_api.py
"""

import json
from app import create_app

# --- Configuración del cliente de prueba -------------------------------------
app = create_app()
client = app.test_client()


def test_health_endpoint():
    """
    Prueba el endpoint ``GET /`` de la API.

    Verifica que:
    - El código de respuesta sea 200 (OK).
    - El JSON devuelto contenga las claves esperadas.

    Returns
    -------
    None
    """
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_endpoint():
    """
    Prueba el endpoint ``POST /predict`` con una instancia válida.

    Envía un JSON con las características del dataset Breast Cancer y
    verifica que el servicio devuelva una predicción válida.

    Returns
    -------
    None
    """
    features = [
        17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
        1.095, 0.9053, 8.589, 153.4, 0.0064, 0.049, 0.0537, 0.0159, 0.03, 0.00619,
        25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189,
    ]

    resp = client.post(
        "/predict",
        data=json.dumps({"features": features}),
        content_type="application/json",
    )

    assert resp.status_code == 200
    data = resp.get_json()
    assert "predictions" in data
    assert "classes" in data
