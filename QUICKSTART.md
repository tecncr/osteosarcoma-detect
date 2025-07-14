# 🚀 Guía de Inicio Rápido

## ✅ Estado del Proyecto
✅ Aplicación Streamlit lista y funcional  
✅ 4 modelos CNN pre-entrenados cargados  
✅ Interfaz responsiva y moderna  
✅ Generación de reportes PDF  
✅ Análisis estadístico completo  

## 🏃‍♂️ Ejecutar la Aplicación

### Opción 1: Usar el script automático
```bash
./setup.sh
source venv/bin/activate
streamlit run app.py
```

### Opción 2: Instalación manual
```bash
# 1. Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar aplicación
streamlit run app.py
```

## 🌐 Acceder a la Aplicación
Una vez ejecutada, la aplicación estará disponible en:
- **Local**: http://localhost:8501
- **Red**: http://IP_DE_TU_MAQUINA:8501

## 🧪 Probar la Aplicación

### Imagen de Prueba
Hemos incluido una imagen de muestra para probar la funcionalidad:
- **Ubicación**: `examples/muestra_histopatologica.png`
- **Uso**: Carga esta imagen en la aplicación para ver todos los resultados

### Generar Nueva Imagen de Prueba
```bash
python create_sample.py
```

## 📊 Características Principales

### 🤖 Modelos de IA Incluidos
1. **VGG16** - Arquitectura clásica profunda
2. **ResNet50** - Red residual avanzada  
3. **MobileNetV2** - Modelo eficiente y ligero
4. **EfficientNetB0** - Balance óptimo precisión/eficiencia

### 📈 Análisis Disponibles
- ✅ Predicciones individuales por modelo
- ✅ Distribución de probabilidades
- ✅ Clasificación consenso
- ✅ Análisis de concordancia entre modelos
- ✅ Distancias entre distribuciones de probabilidad
- ✅ Métricas estadísticas detalladas
- ✅ Interpretación clínica
- ✅ Recomendaciones médicas

### 📄 Exportación
- ✅ Reportes PDF completos con todos los análisis
- ✅ Inclusión de imagen analizada
- ✅ Interpretación clínica profesional

## 🎯 Cómo Usar

1. **📤 Cargar Imagen**: Sube una imagen histopatológica (PNG, JPG, JPEG, TIFF, BMP)
2. **🔄 Procesamiento**: La imagen se redimensiona automáticamente a 224×224 píxeles
3. **🧠 Análisis**: Los 4 modelos realizan inferencia simultáneamente
4. **📊 Resultados**: Visualiza predicciones, probabilidades y concordancia
5. **📄 Exportar**: Genera reporte PDF profesional

## 🏥 Tipos de Diagnóstico

| Clase | Descripción | Recomendación |
|-------|-------------|---------------|
| **Non-Tumor** | Tejido sin presencia de tumor | Seguimiento rutinario |
| **Non-Viable-Tumor** | Tejido tumoral no viable (necrótico) | Evaluar respuesta al tratamiento |
| **Viable** | Tejido tumoral viable (activo) | Considerar tratamiento activo |
| **Mixed** | Tejido mixto con características combinadas | Análisis adicional requerido |

## 🔧 Solución de Problemas

### Error: Modelos no encontrados
```bash
# Verificar que los modelos estén en la carpeta correcta
ls -la models/
# Deberías ver:
# VGG16_osteosarcoma.h5
# ResNet50_osteosarcoma.h5
# MobileNetV2_osteosarcoma.h5
# EfficientNetB0_osteosarcoma.h5
```

### Error: Dependencias faltantes
```bash
pip install --upgrade -r requirements.txt
```

### Error: Puerto ocupado
```bash
# Usar puerto diferente
streamlit run app.py --server.port 8502
```

## 📱 Compatibilidad

- ✅ **Navegadores**: Chrome, Firefox, Safari, Edge
- ✅ **Dispositivos**: Desktop, Tablet, Mobile
- ✅ **Sistemas**: Windows, macOS, Linux
- ✅ **Python**: 3.8+

## ⚠️ Notas Importantes

### 🩺 Uso Médico
- Esta es una **herramienta de apoyo diagnóstico**
- **NO sustituye el juicio clínico profesional**
- Debe ser utilizada por profesionales médicos cualificados
- Los resultados requieren correlación con otros estudios

### 🔒 Seguridad
- No almacena imágenes permanentemente
- Todos los análisis son locales
- Sin conexión a internet requerida después de la instalación

## 📞 Soporte

### Logs de la Aplicación
```bash
# Ver logs en tiempo real
tail -f ~/.streamlit/logs/streamlit.log
```

### Debug Mode
```bash
# Ejecutar en modo debug
streamlit run app.py --logger.level=debug
```

## 🚀 ¡Listo para Usar!

Tu clasificador de osteosarcoma está completamente configurado y listo para uso clínico. La aplicación combina el poder de 4 modelos de deep learning con una interfaz moderna y análisis estadístico completo.

**¡Disfruta analizando imágenes histopatológicas con IA! 🔬✨**
