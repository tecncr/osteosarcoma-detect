# 🔬 Clasificador de Osteosarcoma

Una aplicación web moderna desarrollada con Streamlit para la clasificación automática de imágenes histopatológicas de osteosarcoma utilizando 4 modelos de deep learning pre-entrenados.

## 🎯 Características Principales

- **🤖 Múltiples Modelos de IA**: VGG16, ResNet50, MobileNetV2, EfficientNetB0
- **📊 Análisis Completo**: Predicciones, probabilidades, concordancia entre modelos
- **📈 Visualizaciones Interactivas**: Gráficos dinámicos con Plotly
- **📄 Reportes PDF**: Exportación completa de resultados
- **🎨 UI Responsiva**: Diseño moderno y adaptable
- **🏥 Interpretación Clínica**: Recomendaciones basadas en resultados
- **📱 Optimizado para Móviles**: Funciona en cualquier dispositivo

## 🏥 Clases de Diagnóstico

1. **Non-Tumor**: Tejido sin presencia de tumor
2. **Non-Viable-Tumor**: Tejido tumoral no viable (necrótico)
3. **Viable**: Tejido tumoral viable (activo)
4. **Mixed**: Tejido mixto con características combinadas

## 🚀 Instalación y Uso

### Prerequisitos

- Python 3.8 o superior
- Los modelos pre-entrenados en la carpeta `models/`:
  - `VGG16_osteosarcoma.h5`
  - `ResNet50_osteosarcoma.h5`
  - `MobileNetV2_osteosarcoma.h5`
  - `EfficientNetB0_osteosarcoma.h5`

Puedes descargar los modelos desde [Hugging Face](https://huggingface.co/tecncr/osteosarcoma-detect) o entrenarlos tú mismo siguiendo las instrucciones en el notebook de entrenamiento.

### Instalación

1. **Clonar o descargar el proyecto**
```bash
cd osteosarcoma-detect
```

2. **Crear entorno virtual (recomendado)**
```bash
python -m venv venv

# En Linux/Mac:
source venv/bin/activate

# En Windows:
venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

### Ejecutar la Aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 📱 Cómo Usar la Aplicación

1. **📤 Cargar Imagen**: Sube una imagen histopatológica (PNG, JPG, JPEG, TIFF, BMP)
2. **🔍 Análisis Automático**: La imagen se redimensiona a 224×224 y se analiza con los 4 modelos
3. **📊 Revisar Resultados**: 
   - Clasificación consenso
   - Predicciones individuales por modelo
   - Distribución de probabilidades
   - Análisis de concordancia
4. **📄 Exportar**: Genera un reporte PDF completo con todos los resultados

## 📊 Métricas y Análisis Incluidos

### 🎯 Métricas de Clasificación
- Predicciones por modelo
- Distribución de probabilidades
- Clasificación consenso
- Confianza promedio

### 🤝 Análisis de Concordancia
- Concordancia entre pares de modelos
- Concordancia total de todos los modelos
- Análisis de discrepancias

### 📏 Distancias entre Modelos
- **Distancia Euclidiana**: Distancia geométrica entre vectores de probabilidad
- **Distancia Coseno**: Similitud angular entre vectores
- **Divergencia KL**: Diferencia entre distribuciones de probabilidad

### 📈 Visualizaciones
- Gráficos de barras comparativos
- Mapas de calor de probabilidades
- Análisis de concordancia
- Gráficos de distancias entre modelos

## 📄 Reporte PDF

El reporte PDF incluye:
- ✅ Información del análisis
- 🖼️ Imagen analizada
- 📊 Resultados detallados por modelo
- 🤝 Análisis de concordancia
- 📏 Métricas de distancia
- 🏥 Interpretación clínica
- 💊 Recomendaciones médicas
- ⚠️ Disclaimer médico

## 🔧 Estructura del Proyecto

```
osteosarcoma-detect/
├── app.py                          # Aplicación principal de Streamlit
├── requirements.txt                # Dependencias de Python
├── README.md                      # Documentación
├── models/                        # Modelos pre-entrenados
│   ├── VGG16_osteosarcoma.h5
│   ├── ResNet50_osteosarcoma.h5
│   ├── MobileNetV2_osteosarcoma.h5
│   └── EfficientNetB0_osteosarcoma.h5
└── training/                      # Código de entrenamiento (referencia)
    ├── Osteosarcoma_Entrenamiento_TF_4m.ipynb
    └── Dataset/
```

## 🧠 Modelos de IA Utilizados

### 1. VGG16
- **Arquitectura**: Convolucional profunda clásica
- **Características**: 16 capas, robusto para clasificación de imágenes
- **Preprocesamiento**: Normalización específica de VGG

### 2. ResNet50
- **Arquitectura**: Red residual con conexiones skip
- **Características**: 50 capas, soluciona el problema del gradiente desvaneciente
- **Preprocesamiento**: Normalización específica de ResNet

### 3. MobileNetV2
- **Arquitectura**: Eficiente para dispositivos móviles
- **Características**: Depthwise separable convolutions, menor tamaño
- **Preprocesamiento**: Normalización específica de MobileNet

### 4. EfficientNetB0
- **Arquitectura**: Balance óptimo entre precisión y eficiencia
- **Características**: Compound scaling, estado del arte
- **Preprocesamiento**: Normalización específica de EfficientNet

## ⚠️ Consideraciones Médicas Importantes

### 🩺 Uso Clínico
- Esta es una **herramienta de apoyo diagnóstico**, no un diagnóstico definitivo
- Debe ser utilizada exclusivamente por profesionales médicos cualificados
- Los resultados deben correlacionarse con otros estudios clínicos

### 🔬 Interpretación de Resultados
- **Alta concordancia**: Todos los modelos predicen la misma clase
- **Concordancia parcial**: Algunos modelos difieren, requiere análisis adicional
- **Baja confianza**: Resultados inciertos, considerar estudios complementarios

### 📋 Recomendaciones Clínicas
- **Non-Tumor**: Seguimiento rutinario según protocolo
- **Non-Viable-Tumor**: Evaluar respuesta al tratamiento
- **Viable**: Considerar tratamiento activo inmediato
- **Mixed**: Análisis histopatológico adicional requerido

## 🛠️ Desarrollo y Personalización

### Modificar Clases
Para añadir o modificar clases de diagnóstico, edita las variables en `app.py`:
```python
CLASS_NAMES = ['Non-Tumor', 'Non-Viable-Tumor', 'Viable', 'Mixed']
CLASS_DESCRIPTIONS = {
    # Añadir descripciones aquí
}
```

### Añadir Nuevos Modelos
1. Coloca el archivo `.h5` en la carpeta `models/`
2. Añade la configuración en la función `load_models()`
3. Define la función de preprocesamiento apropiada

### Personalizar Visualizaciones
Las funciones de visualización están en:
- `create_prediction_visualization()`
- `create_agreement_visualization()`

## 📞 Soporte y Contribuciones

Para reportar problemas, sugerir mejoras o contribuir al proyecto:
- Crea un issue con descripción detallada
- Incluye capturas de pantalla si es relevante
- Proporciona información del entorno (Python, SO, etc.)

---

**⚠️ DISCLAIMER MÉDICO**: Esta aplicación es una herramienta de apoyo diagnóstico y NO constituye un diagnóstico médico definitivo. Siempre consulte con profesionales médicos cualificados para interpretación y decisiones clínicas.
