# Clasificación de Géneros Musicales

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Orange)
![Librosa](https://img.shields.io/badge/Librosa-Audio_Processing-yellowgreen)

Este proyecto desarrolla un pipeline de Machine Learning *end-to-end* para la extracción de características y clasificación automática de pistas de audio en 7 géneros musicales diferentes. El estudio abarca desde el preprocesamiento de la señal de audio hasta la evaluación de modelos clásicos y arquitecturas de Deep Learning.

## Tabla de Contenidos
- [Descripción del Proyecto](#descripción-del-proyecto)
- [Dataset y Características](#dataset-y-características)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Metodología y Modelos](#metodología-y-modelos)
- [Resultados Principales](#resultados-principales)
- [Instalación y Requisitos](#instalación-y-requisitos)
- [Trabajo Futuro](#trabajo-futuro)
- [Autores](#autores)

---

## Descripción del Proyecto

El objetivo de este proyecto es asignar automáticamente el género musical a una canción a partir de sus características espectrales y rítmicas. Se compara el desempeño de algoritmos tradicionales (Regresión Logística, Support Vector Machines, ExtraTrees) frente a un modelo de Redes Neuronales (Multi-Layer Perceptron) construido en PyTorch.

**Géneros evaluados (7):** Rock, Classical, Blues, Disco, Country, Pop, y Salsa.

---

## Dataset y Características

El conjunto de datos consta de **937 canciones**. Se utilizó el clásico dataset GTZAN como base, extendido con canciones obtenidas manualmente para mejorar el desempeño y realismo del modelo.

A través de la librería `librosa`, cada audio fue muestreado aleatoriamente en fragmentos de 30 segundos, de los cuales se extrajeron **57 características numéricas** globales (medias y varianzas):
* **MFCCs** (Mel-Frequency Cepstral Coefficients)
* **Chroma STFT** (Información armónica)
* **RMS** (Intensidad de la señal)
* **Spectral Centroid & Rolloff** (Brillo y energía)
* **Zero Crossing Rate (ZCR)**
* **Harmonic & Percussive Components**
* **Tempo** (BPM)

---

## Estructura del Proyecto

```text
 Proyecto_ML
 ┣ genres_original/            # (Opcional) Audios en bruto segmentados por género
 ┣ Extracion_datos.ipynb       # Extracción de audio (pydub) y cálculo de features (librosa)
 ┣ ModeloBaseyAdicionales.ipynb# EDA (t-SNE/UMAP) y modelos Base (LogReg, SVM, PyTorch MLP)
 ┣ ModeloModerno.ipynb         # Limpieza y entrenamiento del ExtraTreesClassifier
 ┣ song_features.csv           # Dataset estructurado con las 57 características tabuladas
 ┣ DocumentoFinalML.pdf        # Informe completo del proyecto (formato paper IEEE)
 ┣ PresentacionFinalML.pdf     # Diapositivas de la presentación del proyecto
 ┗ README.md                   # Este documento
```

---

## Metodología y Modelos

El flujo de trabajo aplicado consta de las siguientes fases:
1. **Normalización:** Uso de `StandardScaler` (para PCA/LDA y modelos lineales) y `MinMaxScaler` (para estabilizar los gradientes en redes neuronales).
2. **Reducción de Dimensionalidad:** PCA (preservando el 95% de varianza), LDA, así como t-SNE y UMAP (exclusivamente para visualización exploratoria).
3. **Modelado:**
   * **Regresión Logística + LDA:** Modelo base valorado por su simplicidad e interpretabilidad.
   * **SVM (Kernel RBF) + PCA:** Identificación de fronteras de decisión no lineales en un subespacio de alta dimensionalidad.
   * **ExtraTreesClassifier:** Ensemble basado en árboles de decisión extremadamente aleatorizados.
   * **MLP (PyTorch):** Red neuronal profunda con múltiples capas densas, *Batch Normalization*, *Dropout* progresivo, optimizador *AdamW* y *Early Stopping*.

---

## Resultados Principales

El problema se evaluó utilizando las métricas de **Accuracy** y **F1-Macro** (dada la naturaleza multiclase del problema con datos balanceados).

| Modelo | Precision | Recall | F1-Macro |
|:---:|:---:|:---:|:---:|
| Regresión Logística + LDA | 0.74 | 0.73 | 0.73 |
| SVM + PCA | 0.77 | 0.77 | 0.77 |
| **MLP (PyTorch)** | **0.82** | **0.81** | **0.81** |
| ExtraTreesClassifier | 0.76 | 0.76 | 0.76 |

* **Conclusión clave:** El modelo MLP demostró la mayor capacidad para capturar patrones complejos, logrando el mejor desempeño global. A través de las matrices de confusión (análisis de errores), se observó que el mayor desafío es el solapamiento acústico natural entre géneros como *rock*, *disco* y *blues*.

---

## Instalación y Requisitos

Para replicar el entorno de desarrollo y ejecutar los notebooks, asegúrate de tener **Python 3.8+** instalado.

1. Clona el repositorio:
   ```bash
   git clone <url_de_tu_repositorio>
   cd Proyecto_ML
   ```

2. Instala las dependencias necesarias:
   ```bash
   pip install librosa pydub numpy pandas matplotlib scikit-learn umap-learn torch
   ```

3. **Nota sobre Audio:** Es posible que necesites tener `ffmpeg` instalado a nivel de sistema operativo para que librerías como `pydub` y `librosa` puedan procesar archivos mp3 correctamente.

---

## Trabajo Futuro

El uso de descriptores tradicionales resume el audio a métricas globales pero compromete información temporal importante. Para futuras iteraciones:
* **Espectrogramas de Mel:** Extraer representaciones visuales bidimensionales del audio.
* **Deep Learning Visual:** Implementar **Redes Neuronales Convolucionales (CNN)** para capturar simultáneamente las dependencias temporales y frecuenciales del timbre instrumental.

---

## Autores

**Universidad de Ingeniería y Tecnología (UTEC)**
* **Patrick Ricardo Medina Reyes** - *Ciencia de Datos*
* **Hector Miguel Espinoza Torres** - *Ciencia de Datos*
* **Franco Arturo La Rosa Cabrejos** - *Ingeniería Civil*
* **Juan Diego Jimenez Quispe** - *Ingeniería Civil*
