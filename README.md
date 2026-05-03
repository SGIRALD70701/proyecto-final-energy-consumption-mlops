# Predicción de Consumo Energético con MLOps

## Descripción

Este proyecto implementa un pipeline de Machine Learning para la predicción del consumo energético de un hogar, utilizando el dataset **Individual Household Electric Power Consumption** (UCI).

Se integran buenas prácticas de **MLOps**, incluyendo:

* Seguimiento de experimentos con MLflow
* Automatización con GitHub Actions (CI/CD)
* Reproducibilidad mediante `requirements.txt` y configuración externa

---

## Dataset

El dataset utilizado en este proyecto corresponde al conjunto de datos “Individual Household Electric Power Consumption”, disponible públicamente en el repositorio de UCI: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

Este conjunto de datos contiene mediciones del consumo eléctrico de un hogar con una frecuencia de muestreo de un minuto, recopiladas durante aproximadamente 4 años (desde diciembre de 2006 hasta noviembre de 2010), con más de 2 millones de registros.

Características principales del dataset:
Tipo de datos: multivariado, series de tiempo
Número de registros: 2,075,259
Número de variables: 9
Problemas asociados: regresión y clustering
Variables incluidas:
Global_active_power: consumo activo global (variable objetivo)
Global_reactive_power: consumo reactivo
Voltage: voltaje
Global_intensity: intensidad de corriente
Sub_metering_1,2,3: consumo por zonas específicas del hogar
Date y Time: variables temporales

El dataset presenta aproximadamente un 1.25% de valores faltantes, lo cual fue considerado durante el proceso de preprocesamiento de datos.

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/SGIRALD70701/proyecto-final-energy-consumption-mlops.git
cd proyecto-final-energy-consption-mlops
```

### 2. Crear entorno virtual

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Uso

### Ejecutar el entrenamiento del modelo

```bash
python src/train.py
```

Este comando:

* Carga y limpia los datos
* Entrena el modelo XGBoost
* Calcula métricas
* Registra el experimento en MLflow
* Guarda el modelo como artefacto

---

## Resultados

El modelo obtuvo las siguientes métricas:

* **MSE:** 0.000929
* **RMSE:** 0.03048
* **MAE:** 0.01835
* **R²:** 0.99917

Estos resultados indican un alto desempeño en la predicción del consumo energético.

---

## MLflow

Se utilizó MLflow para registrar:

* Parámetros del modelo
* Métricas de evaluación
* Modelo entrenado
* Ejemplo de entrada

### Visualización

Ejecutar:

```bash
mlflow ui
```

Abrir en el navegador:

```
http://127.0.0.1:5000
```

### Registro del experimento en MLflow**
<img width="960" height="511" alt="image" src="https://github.com/user-attachments/assets/ec81d520-3e92-47e8-a9fc-db9f036b9937" />



---

## CI/CD con GitHub Actions

El pipeline automatizado incluye:

* Clonación del repositorio
* Instalación de dependencias
* Validación del código
* Ejecución de pruebas
* Entrenamiento del modelo
* Subida de artefactos

### Ejecución del pipeline en GitHub Actions**
<img width="960" height="496" alt="image" src="https://github.com/user-attachments/assets/be4725b7-0c3c-4472-a9c9-09abd1591227" />



Ver ejecución:
https://github.com/SGIRALD70701/proyecto-final-energy-consumption-mlops/actions

---

## Estructura del Proyecto

```
proyecto-final-energy-consumption-mlops/
│
├── src/
│   ├── train.py
│   └── test_train.py
│
├── data/
│
├── .github/workflows/
│   └── ml.yml
│
├── config.yaml
├── requirements.txt
├── Makefile
└── README.md
```
