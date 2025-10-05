# Actividad Modular - Kibernum Academy - MLOps / Machine Learning

## Descripción
Proyecto final del módulo 10.  
Implementa un flujo completo de **entrenamiento de modelo**, **API Flask**, **tests automatizados** y **despliegue en Docker**.

El modelo usa datos de diagnóstico de cáncer de mama (Breast Cancer Wisconsin dataset) y realiza predicciones sobre si un caso es *maligno* o *benigno*.

---

## Estructura del proyecto
```
.
├── app.py                  # API Flask con endpoints /
├── train_model.py          # Entrenamiento y serialización del modelo
├── logger.py               # Logger centralizado
├── model.joblib            # Modelo entrenado
├── test_api.py             # Test de ping y predicción
├── test_api_errors.py      # Test de errores en la API
├── requirements.txt        # Dependencias del entorno
├── Dockerfile              # Imagen de despliegue
├── docker-compose.yml      # Configuración de servicio
├── ping.json               # Ejemplo respuesta / (ping)
├── predict.json            # Ejemplo respuesta /predict
└── README.md
```

---

## Requisitos

- Python 3.12  
- scikit-learn 1.7.2  
- Docker y Docker Compose  
- pytest  

Instalación local:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Entrenamiento del modelo
```bash
python train_model.py
```
Genera `model.joblib` con un **RandomForestClassifier**.

---

## Ejecución de la API local
```bash
python app.py
```

Prueba endpoints:
```bash
# Ping
curl http://127.0.0.1:8000/

# Predicción
curl -X POST http://127.0.0.1:8000/predict   -H "Content-Type: application/json"   -d '{"features":[17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,
                   1.095,0.9053,8.589,153.4,0.0064,0.049,0.0537,0.0159,0.03,0.00619,
                   25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]}'
```

Ejemplo de respuesta:
```json
{"classes":["malignant"],"predictions":[0],"probabilities":[[0.96,0.04]]}
```

---

## Ejecución con Docker

Construir imagen:
```bash
docker build -t mlops-bc:latest .
```

Ejecutar contenedor:
```bash
docker run -d --name mlops-bc -p 8000:8000 mlops-bc:latest
```

Probar:
```bash
curl http://127.0.0.1:8000/
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"features":[17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.0064,0.049,0.0537,0.0159,0.03,0.00619,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]}'
```

---

## Tests automatizados
```bash
pytest -q
```

---

## Autor
**Walter M.**  
Repositorio: [kibernum_ml_modulo_10_actividad_modular](https://github.com/walterm2482/kibernum_ml_modulo_10_actividad_modular)
