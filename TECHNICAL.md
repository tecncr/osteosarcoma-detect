# 🛠️ Documentación Técnica - Clasificador de Osteosarcoma

## 📋 Arquitectura del Sistema

### 🧠 Modelos de Deep Learning

#### 1. VGG16 (Visual Geometry Group)
```python
# Configuración del modelo
- Capas: 16 capas profundas
- Parámetros: ~138M
- Input: (224, 224, 3)
- Preprocesamiento: mean=[123.68, 116.779, 103.939]
- Transfer Learning: ImageNet → Osteosarcoma
```

#### 2. ResNet50 (Residual Network)
```python
# Configuración del modelo
- Capas: 50 capas con conexiones residuales
- Parámetros: ~25M
- Input: (224, 224, 3)
- Preprocesamiento: mean=[123.68, 116.779, 103.939]
- Ventaja: Soluciona gradiente desvaneciente
```

#### 3. MobileNetV2 (Mobile Networks)
```python
# Configuración del modelo
- Capas: Depthwise separable convolutions
- Parámetros: ~3.5M
- Input: (224, 224, 3)
- Preprocesamiento: mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]
- Ventaja: Eficiencia computacional
```

#### 4. EfficientNetB0 (Efficient Networks)
```python
# Configuración del modelo
- Capas: Compound scaling method
- Parámetros: ~5.3M
- Input: (224, 224, 3)
- Preprocesamiento: mean=[123.68, 116.779, 103.939]
- Ventaja: Balance óptimo precisión/eficiencia
```

### 🏗️ Arquitectura de la Aplicación

```
app.py
├── 🎨 UI Components (Streamlit)
│   ├── Header & Navigation
│   ├── File Upload
│   ├── Image Display
│   └── Results Dashboard
│
├── 🤖 AI Pipeline
│   ├── Image Preprocessing
│   ├── Model Loading (@st.cache_resource)
│   ├── Batch Inference
│   └── Post-processing
│
├── 📊 Analytics Engine
│   ├── Prediction Analysis
│   ├── Concordance Metrics
│   ├── Probability Distances
│   └── Statistical Tests
│
└── 📄 Report Generation
    ├── PDF Creation (ReportLab)
    ├── Charts & Visualizations
    └── Clinical Interpretation
```

## 🔧 Funciones Principales

### 🖼️ Procesamiento de Imágenes

```python
def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocesa imagen para inferencia
    
    Args:
        image (PIL.Image): Imagen original
        target_size (tuple): Tamaño objetivo (224, 224)
    
    Returns:
        np.ndarray: Array de imagen preprocesada
    """
```

### 🧠 Inferencia con Modelos

```python
def predict_with_models(image, models):
    """
    Realiza inferencia con todos los modelos
    
    Args:
        image (PIL.Image): Imagen a analizar
        models (dict): Diccionario de modelos cargados
    
    Returns:
        tuple: (predictions, probabilities)
    """
```

### 📊 Análisis de Concordancia

```python
def calculate_agreement_metrics(predictions):
    """
    Calcula métricas de concordancia entre modelos
    
    Args:
        predictions (dict): Predicciones de cada modelo
    
    Returns:
        dict: Métricas de concordancia
    """
```

### 📏 Distancias de Probabilidad

```python
def calculate_probability_distances(probabilities):
    """
    Calcula distancias entre distribuciones
    
    Métricas incluidas:
    - Distancia Euclidiana
    - Distancia Coseno  
    - Divergencia KL
    """
```

## 📈 Visualizaciones

### 🎯 Gráficos de Predicción

```python
# Gráfico de barras comparativo
fig1 = px.bar(
    df_probs, 
    x='Clase', 
    y='Probabilidad', 
    color='Modelo',
    title='Distribución de Probabilidades por Modelo'
)

# Heatmap de probabilidades
fig2 = px.imshow(
    prob_matrix.values,
    title='Mapa de Calor - Probabilidades por Modelo'
)
```

### 🤝 Análisis de Concordancia

```python
# Gráfico de concordancia
fig_agreement = px.bar(
    df_agreement,
    x='Comparación',
    color='Concordancia',
    title='Concordancia entre Modelos'
)

# Gráfico de distancias
fig_distances = make_subplots(
    rows=1, cols=3,
    subplot_titles=('Euclidiana', 'Coseno', 'KL')
)
```

## 📄 Generación de PDF

### 🏗️ Estructura del Reporte

```python
def generate_pdf_report(image, predictions, probabilities, 
                       agreement_metrics, probability_distances):
    """
    Estructura del PDF:
    1. Portada y metadatos
    2. Imagen analizada
    3. Resultados por modelo
    4. Análisis de concordancia
    5. Métricas estadísticas
    6. Interpretación clínica
    7. Recomendaciones
    8. Disclaimer médico
    """
```

### 📊 Elementos del PDF

- **Texto**: Párrafos con estilos personalizados
- **Imágenes**: Imagen analizada redimensionada
- **Tablas**: Resultados estructurados
- **Gráficos**: (Futuro: integración con matplotlib)

## 🎨 Diseño UI/UX

### 🎨 CSS Personalizado

```css
/* Colores principales */
--primary-color: #2E86AB;
--secondary-color: #A23B72;
--background-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

/* Componentes */
.main-header { /* Encabezado principal */ }
.model-card { /* Tarjetas de modelos */ }
.metric-card { /* Métricas destacadas */ }
.info-box { /* Cajas informativas */ }
```

### 📱 Responsividad

```python
# Layout responsivo con columnas
col1, col2 = st.columns([1, 2])  # Proporción 1:2
col1, col2, col3 = st.columns(3)  # Tres columnas iguales

# Gráficos adaptativos
st.plotly_chart(fig, use_container_width=True)
```

## 🔧 Optimizaciones de Rendimiento

### 💾 Caché de Modelos

```python
@st.cache_resource
def load_models():
    """Carga modelos una sola vez y los mantiene en memoria"""
    # Reduce tiempo de carga significativamente
    # Memoria compartida entre sesiones
```

### ⚡ Procesamiento Paralelo

```python
# Inferencia simultánea (futuro)
with ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(model.predict, processed_image)
        for model in models.values()
    ]
    results = [f.result() for f in futures]
```

### 🧹 Gestión de Memoria

```python
# Liberación de memoria GPU
def clear_memory():
    gc.collect()
    tf.keras.backend.clear_session()
```

## 📊 Métricas Implementadas

### 🎯 Métricas de Clasificación

| Métrica | Fórmula | Interpretación |
|---------|---------|----------------|
| **Accuracy** | TP+TN / Total | Precisión general |
| **Precision** | TP / (TP+FP) | Precisión por clase |
| **Recall** | TP / (TP+FN) | Sensibilidad |
| **F1-Score** | 2×(P×R)/(P+R) | Balance P/R |

### 🤝 Métricas de Concordancia

```python
# Concordancia simple
agreement = (pred1 == pred2)

# Kappa de Cohen (futuro)
kappa = cohen_kappa_score(pred1, pred2)
```

### 📏 Distancias de Probabilidad

```python
# Distancia Euclidiana
euclidean_dist = np.sqrt(np.sum((p1 - p2)**2))

# Distancia Coseno
cosine_dist = 1 - np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))

# Divergencia KL
kl_div = np.sum(p1 * np.log(p1 / p2))
```

## 🔒 Seguridad y Privacidad

### 🛡️ Medidas de Seguridad

```python
# Validación de archivos
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'bmp'}

# Limitación de tamaño
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Sanitización de entradas
def validate_image(uploaded_file):
    # Verificar formato, tamaño, contenido
```

### 🔐 Privacidad

- **Sin almacenamiento permanente** de imágenes
- **Procesamiento local** sin envío a servidores externos
- **Sesiones aisladas** entre usuarios
- **Limpieza automática** de memoria

## 🚀 Extensibilidad

### ➕ Agregar Nuevos Modelos

```python
# 1. Añadir modelo a la carpeta models/
# 2. Definir función de preprocesamiento
# 3. Actualizar load_models()

def create_new_model(input_shape, num_classes):
    # Definir arquitectura
    # Compilar modelo
    # Retornar modelo configurado
```

### 📊 Nuevas Métricas

```python
# Ejemplo: Agregar métrica personalizada
def calculate_custom_metric(predictions, ground_truth):
    # Implementar lógica de la métrica
    return metric_value

# Integrar en el pipeline de análisis
```

### 🎨 Personalización UI

```python
# Temas personalizados
def apply_custom_theme():
    st.markdown("""
    <style>
    /* CSS personalizado */
    </style>
    """, unsafe_allow_html=True)
```

## 🧪 Testing y Validación

### 🔬 Tests Unitarios (Futuro)

```python
import pytest

def test_image_preprocessing():
    # Test redimensionamiento
    # Test normalización
    # Test formato de salida

def test_model_inference():
    # Test predicciones válidas
    # Test shape de output
    # Test valores de probabilidad

def test_metrics_calculation():
    # Test métricas de concordancia
    # Test distancias de probabilidad
    # Test edge cases
```

### ✅ Validación de Modelos

```python
def validate_model_output(predictions, probabilities):
    """
    Valida que las salidas de los modelos sean correctas:
    - Predicciones en rango válido [0, num_classes-1]
    - Probabilidades suman 1.0
    - No valores NaN o infinitos
    """
```

## 📦 Deployment

### 🐳 Docker (Futuro)

```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

### ☁️ Cloud Deployment

```bash
# Heroku
git push heroku main

# AWS/GCP
# Configurar container registry
# Deploy con terraform/ansible
```

## 📈 Monitoreo y Logs

### 📊 Métricas de Uso (Futuro)

```python
# Logging de eventos
import logging

logging.info(f"Prediction made: {model_name} -> {prediction}")
logging.info(f"Processing time: {elapsed_time:.2f}s")

# Métricas de performance
@st.cache_data
def log_performance_metrics():
    # Tiempo de inferencia por modelo
    # Uso de memoria
    # Distribución de predicciones
```

## 🎯 Roadmap Futuro

### 🔮 Mejoras Planificadas

1. **🧠 AI/ML**
   - [ ] Ensemble learning
   - [ ] Uncertainty quantification
   - [ ] Active learning
   - [ ] Model explainability (LIME/SHAP)

2. **🎨 UI/UX**
   - [ ] Modo oscuro
   - [ ] Comparación side-by-side
   - [ ] Histórico de análisis
   - [ ] Exportación múltiple

3. **📊 Analytics**
   - [ ] Dashboard de estadísticas
   - [ ] Análisis de tendencias
   - [ ] Alertas automáticas
   - [ ] Métricas avanzadas

4. **🔧 Técnico**
   - [ ] API REST
   - [ ] Base de datos
   - [ ] Autenticación
   - [ ] Escalabilidad horizontal

## 📚 Referencias

### 📖 Papers Científicos
- VGG: "Very Deep Convolutional Networks for Large-Scale Image Recognition"
- ResNet: "Deep Residual Learning for Image Recognition"
- MobileNet: "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
- EfficientNet: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"

### 🛠️ Tecnologías Utilizadas
- **Frontend**: Streamlit, Plotly, CSS3
- **Backend**: Python, TensorFlow/Keras
- **AI/ML**: Transfer Learning, CNN, Computer Vision
- **Análisis**: NumPy, Pandas, SciPy, Scikit-learn
- **Visualización**: Matplotlib, Seaborn, Plotly
- **PDF**: ReportLab
- **Deployment**: Docker (futuro), Cloud platforms

---

**🚀 Esta documentación técnica proporciona una visión completa de la arquitectura, implementación y posibilidades de extensión del clasificador de osteosarcoma.**
