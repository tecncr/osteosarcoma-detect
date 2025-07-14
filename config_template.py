# ⚙️ Configuración del Clasificador de Osteosarcoma

# Este archivo permite personalizar varios aspectos de la aplicación
# Copia este archivo como config.py y modifica los valores según necesites

import os

# =============================================================================
# 🎨 CONFIGURACIÓN DE INTERFAZ
# =============================================================================

# Colores del tema
THEME_CONFIG = {
    "primary_color": "#2E86AB",
    "secondary_color": "#A23B72", 
    "background_color": "#FFFFFF",
    "text_color": "#262730",
    "accent_color": "#667eea"
}

# Títulos y textos
APP_CONFIG = {
    "title": "🔬 Clasificador de Osteosarcoma",
    "subtitle": "Análisis automatizado de imágenes histopatológicas",
    "description": "Sistema de apoyo diagnóstico basado en IA para la clasificación de osteosarcoma",
    "footer": "Desarrollado para uso académico y de investigación médica"
}

# =============================================================================
# 🤖 CONFIGURACIÓN DE MODELOS
# =============================================================================

# Directorio de modelos
MODELS_DIR = "models"

# Configuración de modelos disponibles
MODELS_CONFIG = {
    "VGG16": {
        "file": "VGG16_osteosarcoma.h5",
        "name": "VGG16",
        "description": "Arquitectura clásica con capas convolucionales profundas",
        "color": "#FF6B6B",
        "enabled": True
    },
    "ResNet50": {
        "file": "ResNet50_osteosarcoma.h5", 
        "name": "ResNet50",
        "description": "Red residual con conexiones skip para mejor entrenamiento",
        "color": "#4ECDC4",
        "enabled": True
    },
    "MobileNetV2": {
        "file": "MobileNetV2_osteosarcoma.h5",
        "name": "MobileNetV2", 
        "description": "Arquitectura eficiente para dispositivos móviles",
        "color": "#45B7D1",
        "enabled": True
    },
    "EfficientNetB0": {
        "file": "EfficientNetB0_osteosarcoma.h5",
        "name": "EfficientNetB0",
        "description": "Modelo balanceado entre precisión y eficiencia", 
        "color": "#96CEB4",
        "enabled": True
    }
}

# =============================================================================
# 🏥 CONFIGURACIÓN MÉDICA  
# =============================================================================

# Clases de diagnóstico
CLASS_CONFIG = {
    "names": ["Non-Tumor", "Non-Viable-Tumor", "Viable", "Mixed"],
    "descriptions": {
        "Non-Tumor": "Tejido sin presencia de tumor",
        "Non-Viable-Tumor": "Tejido tumoral no viable (necrótico)", 
        "Viable": "Tejido tumoral viable (activo)",
        "Mixed": "Tejido mixto con características combinadas"
    },
    "colors": {
        "Non-Tumor": "#28a745",
        "Non-Viable-Tumor": "#ffc107", 
        "Viable": "#dc3545",
        "Mixed": "#6f42c1"
    },
    "recommendations": {
        "Non-Tumor": "👍 El análisis sugiere ausencia de tejido tumoral. Se recomienda seguimiento rutinario según protocolo clínico.",
        "Non-Viable-Tumor": "⚡ Se detecta tejido tumoral no viable (necrótico). Evaluar respuesta al tratamiento previo y considerar ajustes terapéuticos.",
        "Viable": "⚠️ Se detecta tejido tumoral viable activo. Considerar opciones de tratamiento inmediato según guidelines oncológicos.", 
        "Mixed": "🔄 Se detecta tejido mixto con características heterogéneas. Se recomienda análisis histopatológico adicional y evaluación multidisciplinaria."
    }
}

# =============================================================================
# 🖼️ CONFIGURACIÓN DE IMÁGENES
# =============================================================================

# Procesamiento de imágenes
IMAGE_CONFIG = {
    "target_size": (224, 224),
    "allowed_formats": ["png", "jpg", "jpeg", "tiff", "bmp"],
    "max_file_size_mb": 10,
    "quality_threshold": 0.7
}

# Validación de imágenes
VALIDATION_CONFIG = {
    "min_resolution": (100, 100),
    "max_resolution": (4096, 4096),
    "validate_medical_format": True,
    "auto_enhance": False
}

# =============================================================================
# 📊 CONFIGURACIÓN DE ANÁLISIS
# =============================================================================

# Métricas de concordancia
AGREEMENT_CONFIG = {
    "calculate_kappa": True,
    "calculate_fleiss_kappa": False,  # Para más de 2 evaluadores
    "agreement_threshold": 0.8,
    "require_unanimous": False
}

# Distancias de probabilidad
DISTANCE_CONFIG = {
    "calculate_euclidean": True,
    "calculate_cosine": True, 
    "calculate_kl_divergence": True,
    "calculate_js_divergence": False,  # Jensen-Shannon
    "calculate_wasserstein": False     # Earth Mover's Distance
}

# Umbrales de confianza
CONFIDENCE_CONFIG = {
    "high_confidence": 0.8,
    "medium_confidence": 0.6,
    "low_confidence": 0.4,
    "uncertainty_threshold": 0.3
}

# =============================================================================
# 📄 CONFIGURACIÓN DE REPORTES
# =============================================================================

# Configuración de PDF
PDF_CONFIG = {
    "page_size": "letter",  # letter, A4, legal
    "margins": (72, 72, 72, 72),  # top, right, bottom, left
    "font_family": "Helvetica",
    "include_logo": False,
    "include_watermark": False,
    "language": "es"  # es, en
}

# Secciones del reporte
REPORT_SECTIONS = {
    "include_executive_summary": True,
    "include_image_analysis": True,
    "include_model_details": True,
    "include_statistical_analysis": True,
    "include_clinical_interpretation": True,
    "include_recommendations": True,
    "include_disclaimer": True,
    "include_technical_details": False
}

# =============================================================================
# 🔧 CONFIGURACIÓN TÉCNICA
# =============================================================================

# Performance
PERFORMANCE_CONFIG = {
    "enable_gpu": True,
    "memory_limit_gb": 4,
    "batch_size": 1,
    "num_threads": 4,
    "cache_models": True
}

# Logging
LOGGING_CONFIG = {
    "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    "log_predictions": True,
    "log_performance": True,
    "log_errors": True,
    "save_logs": True,
    "log_file": "logs/osteosarcoma_classifier.log"
}

# Seguridad
SECURITY_CONFIG = {
    "validate_file_type": True,
    "scan_for_malware": False,
    "encrypt_sensitive_data": False,
    "enable_audit_trail": False,
    "max_sessions": 10
}

# =============================================================================
# 🌐 CONFIGURACIÓN DE DESPLIEGUE
# =============================================================================

# Streamlit
STREAMLIT_CONFIG = {
    "port": 8501,
    "host": "localhost",
    "enable_cors": False,
    "enable_xsrf_protection": False,
    "max_upload_size": 200,  # MB
    "theme_base": "light"
}

# Base de datos (futuro)
DATABASE_CONFIG = {
    "enabled": False,
    "type": "sqlite",  # sqlite, postgresql, mysql
    "host": "localhost",
    "port": 5432,
    "database": "osteosarcoma_db",
    "save_results": False,
    "save_images": False
}

# =============================================================================
# 📈 CONFIGURACIÓN DE MÉTRICAS
# =============================================================================

# Métricas de evaluación
METRICS_CONFIG = {
    "calculate_accuracy": True,
    "calculate_precision": True,
    "calculate_recall": True,
    "calculate_f1_score": True,
    "calculate_auc": True,
    "calculate_mcc": True,
    "calculate_balanced_accuracy": True
}

# Visualizaciones
VISUALIZATION_CONFIG = {
    "plot_height": 500,
    "plot_width": 700,
    "color_palette": "viridis",
    "show_confidence_intervals": True,
    "animation_enabled": False,
    "interactive_plots": True
}

# =============================================================================
# 🚨 CONFIGURACIÓN DE ALERTAS
# =============================================================================

# Alertas médicas
ALERT_CONFIG = {
    "enable_alerts": True,
    "high_risk_threshold": 0.8,
    "uncertainty_alert": True,
    "discordance_alert": True,
    "email_notifications": False,
    "sound_alerts": False
}

# =============================================================================
# 🔄 CONFIGURACIÓN DE ACTUALIZACIONES
# =============================================================================

# Versionado
VERSION_CONFIG = {
    "app_version": "1.0.0",
    "model_version": "2024.1",
    "check_updates": False,
    "auto_update": False,
    "update_url": "https://github.com/tu-repo/osteosarcoma-classifier"
}

# =============================================================================
# EJEMPLO DE USO
# =============================================================================

def get_config():
    """
    Retorna la configuración completa de la aplicación
    
    Returns:
        dict: Configuración completa
    """
    return {
        "theme": THEME_CONFIG,
        "app": APP_CONFIG,
        "models": MODELS_CONFIG,
        "classes": CLASS_CONFIG,
        "images": IMAGE_CONFIG,
        "validation": VALIDATION_CONFIG,
        "agreement": AGREEMENT_CONFIG,
        "distances": DISTANCE_CONFIG,
        "confidence": CONFIDENCE_CONFIG,
        "pdf": PDF_CONFIG,
        "report_sections": REPORT_SECTIONS,
        "performance": PERFORMANCE_CONFIG,
        "logging": LOGGING_CONFIG,
        "security": SECURITY_CONFIG,
        "streamlit": STREAMLIT_CONFIG,
        "database": DATABASE_CONFIG,
        "metrics": METRICS_CONFIG,
        "visualization": VISUALIZATION_CONFIG,
        "alerts": ALERT_CONFIG,
        "version": VERSION_CONFIG
    }

def load_custom_config():
    """
    Carga configuración personalizada si existe
    
    Returns:
        dict: Configuración personalizada o por defecto
    """
    try:
        import config
        return config.get_config()
    except ImportError:
        # Usar configuración por defecto
        return get_config()

# =============================================================================
# INSTRUCCIONES DE USO
# =============================================================================

"""
📝 CÓMO PERSONALIZAR LA APLICACIÓN:

1. 🔧 CONFIGURACIÓN BÁSICA:
   - Copia este archivo como 'config.py'
   - Modifica los valores según tus necesidades
   - Reinicia la aplicación para aplicar cambios

2. 🎨 PERSONALIZAR COLORES:
   - Modifica THEME_CONFIG para cambiar la paleta de colores
   - Los colores deben estar en formato hexadecimal (#RRGGBB)

3. 🤖 HABILITAR/DESHABILITAR MODELOS:
   - Cambia 'enabled': False para deshabilitar un modelo
   - Asegúrate de que el archivo del modelo existe

4. 🏥 AÑADIR NUEVAS CLASES:
   - Actualiza CLASS_CONFIG con nuevas clases
   - Proporciona descripciones y recomendaciones

5. 📊 CONFIGURAR MÉTRICAS:
   - Habilita/deshabilita métricas en METRICS_CONFIG
   - Ajusta umbrales en CONFIDENCE_CONFIG

6. 📄 PERSONALIZAR REPORTES:
   - Modifica PDF_CONFIG para cambiar formato
   - Selecciona secciones en REPORT_SECTIONS

EJEMPLO DE USO EN APP.PY:
```python
from config_template import load_custom_config

config = load_custom_config()
THEME = config['theme']
APP_TITLE = config['app']['title']
```

⚠️ NOTA IMPORTANTE:
- Mantén una copia de respaldo de tu configuración
- Valida los cambios antes de desplegar en producción
- Algunos cambios requieren reiniciar la aplicación
"""
