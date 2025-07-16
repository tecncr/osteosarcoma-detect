# -*- coding: utf-8 -*-
"""
Internationalization module for Osteosarcoma Detection System
Supports English and Spanish languages
"""

# Complete translation dictionaries
TRANSLATIONS = {
    'es': {
        # Page configuration
        'page_title': 'Clasificador de Osteosarcoma',
        'page_icon': '🔬',
        
        # Main headers
        'main_header': '🔬 Clasificador de Osteosarcoma',
        'sub_header_upload': '📤 Cargar Imagen para Análisis',
        'sub_header_results': '📊 Resultados del Análisis',
        
        # Sidebar
        'sidebar_system_info': '📋 Información del Sistema',
        'sidebar_objective': '🎯 Objetivo',
        'sidebar_objective_desc': 'Clasificación automática de imágenes histopatológicas para detección y análisis de osteosarcoma utilizando 4 modelos de deep learning.',
        'sidebar_available_models': '🤖 Modelos Disponibles',
        'sidebar_diagnosis_classes': '🏥 Clases de Diagnóstico',
        'sidebar_language': '🌐 Idioma',
        
        # Model descriptions
        'model_vgg16_desc': 'Arquitectura clásica profunda',
        'model_resnet50_desc': 'Red residual avanzada',
        'model_mobilenetv2_desc': 'Modelo eficiente y ligero',
        'model_efficientnetb0_desc': 'Balance óptimo precisión/eficiencia',
        
        # Class descriptions
        'class_non_tumor': 'Tejido sin presencia de tumor',
        'class_non_viable_tumor': 'Tejido tumoral no viable (necrótico)',
        'class_viable': 'Tejido tumoral viable (activo)',
        'class_mixed': 'Tejido mixto con características combinadas',
        
        # File upload
        'file_upload_label': 'Selecciona una imagen histopatológica',
        'file_upload_help': 'Formatos soportados: PNG, JPG, JPEG, TIFF, BMP',
        'supported_formats': 'Formatos soportados',
        
        # Image info
        'image_info_title': 'Información de la imagen:',
        'image_dimensions': 'Dimensiones',
        'image_mode': 'Modo',
        'image_format': 'Formato',
        'image_size': 'Tamaño',
        'image_original': 'Imagen Original',
        'image_resized': 'Imagen Redimensionada (224×224)',
        'image_resize_note': 'La imagen será redimensionada a 224×224 píxeles para el análisis, manteniendo la calidad necesaria para la clasificación.',
        
        # Buttons
        'btn_start_analysis': '🚀 Iniciar Análisis',
        'btn_new_analysis': '🔄 Nuevo Análisis',
        'btn_generate_pdf': '📥 Generar Reporte PDF',
        'btn_download_pdf': '⬇️ Descargar Reporte PDF',
        
        # Loading messages
        'loading_models': '🔄 Cargando modelos de IA...',
        'loading_analysis': '🔍 Analizando imagen con modelos de IA...',
        'loading_pdf': '📝 Generando reporte PDF...',
        
        # Success messages
        'models_loaded': 'modelos cargados exitosamente',
        'pdf_generated': '✅ Reporte PDF generado exitosamente',
        'file_size': 'Tamaño del archivo',
        
        # Error messages
        'error_loading_models': '❌ No se pudieron cargar los modelos. Verifica que los archivos .h5 estén en la carpeta \'models/\'',
        'error_loading_model': 'Error cargando modelo',
        'error_prediction': 'Error en predicción con',
        'error_inserting_image': 'Error al insertar imagen',
        'error_no_valid_predictions': '❌ No hay predicciones válidas para generar el reporte',
        'error_generating_pdf': '❌ Error al generar el PDF',
        'error_empty_pdf': '❌ Error: El archivo PDF está vacío',
        'error_pdf_buffer': '❌ Error al generar el buffer del PDF',
        'error_reload_suggestion': '💡 Intente recargar la página y volver a subir la imagen',
        'error_show_details': '🔍 Mostrar detalles del error',
        
        # Analysis results
        'consensus_classification': '🎯 Clasificación Consenso',
        'average_confidence': '📈 Confianza Promedio',
        'total_agreement': '🤝 Concordancia Total',
        'agreement_yes': '✅ Sí',
        'agreement_no': '⚠️ No',
        'detailed_results': '🔍 Resultados Detallados por Modelo',
        'prediction': 'Predicción',
        'confidence': 'Confianza',
        'description': 'Descripción',
        'class': 'Clase',
        'probability': 'Probabilidad',
        
        # Visualizations
        'visualizations': '📈 Visualizaciones',
        'probability_distribution': 'Distribución de Probabilidades por Modelo',
        'probability_heatmap': 'Mapa de Calor - Probabilidades por Modelo',
        'model_agreement': 'Concordancia entre Modelos',
        'probability_distances': 'Distancias entre Distribuciones de Probabilidad',
        
        # Statistical analysis
        'statistical_analysis': '📊 Métricas Estadísticas Detalladas',
        'agreement_analysis': '🤝 Análisis de Concordancia entre Modelos',
        'distance_analysis': '📏 Distancias entre Distribuciones de Probabilidad',
        'agreement_concordance': '🔗 Análisis de Concordancia y Distancias',
        
        # Statistical table headers
        'model_1': 'Modelo 1',
        'model_2': 'Modelo 2',
        'prediction_1': 'Predicción 1',
        'prediction_2': 'Predicción 2',
        'concordance': 'Concordancia',
        'euclidean_distance': 'Distancia Euclidiana',
        'cosine_distance': 'Distancia Coseno',
        'kl_divergence': 'Divergencia KL',
        
        # Distance interpretation
        'distance_interpretation': '💡 Interpretación de Distancias:',
        'euclidean_desc': 'Distancia geométrica entre vectores de probabilidad (menor = mayor similitud)',
        'cosine_desc': 'Similitud angular entre vectores (menor = mayor similitud)',
        'kl_desc': 'Medida de diferencia entre distribuciones (menor = mayor similitud)',
        
        # Clinical interpretation
        'clinical_interpretation': '🏥 Interpretación Clínica',
        'diagnostic_summary': '📋 Resumen Diagnóstico',
        'main_classification': 'Clasificación principal',
        'consensus_confidence': 'Confianza del consenso',
        'model_concordance': 'Concordancia entre modelos',
        'concordance_high': 'Alta',
        'concordance_partial': 'Parcial',
        'clinical_recommendation': '💊 Recomendación Clínica',
        
        # Clinical recommendations
        'recommendation_non_tumor': '👍 El análisis sugiere ausencia de tejido tumoral. Se recomienda seguimiento rutinario según protocolo clínico.',
        'recommendation_non_viable': '⚡ Se detecta tejido tumoral no viable (necrótico). Evaluar respuesta al tratamiento previo y considerar ajustes terapéuticos.',
        'recommendation_viable': '⚠️ Se detecta tejido tumoral viable activo. Considerar opciones de tratamiento inmediato según guidelines oncológicos.',
        'recommendation_mixed': '🔄 Se detecta tejido mixto con características heterogéneas. Se recomienda análisis histopatológico adicional y evaluación multidisciplinaria.',
        
        # PDF Report
        'export_report': '📄 Exportar Reporte',
        'pdf_report_title': '🔬 Reporte de Análisis de Osteosarcoma',
        'analysis_info': '📋 Información del Análisis',
        'analysis_date': 'Fecha de análisis',
        'models_used': 'Modelos utilizados',
        'classes_analyzed': 'Clases analizadas',
        'analyzed_image': '🖼️ Imagen Analizada',
        'prediction_results': '🤖 Resultados de Predicción',
        'confidence_label': 'Confianza',
        'probability_distribution_label': 'Distribución de probabilidades',
        'agreement_analysis_pdf': '🤝 Análisis de Concordancia',
        'all_models_agree': '✅ Todos los modelos están de acuerdo en la predicción',
        'models_disagree': '⚠️ Los modelos no concuerdan completamente',
        'pair_details': 'Detalles por pares de modelos',
        'agree': 'Concuerdan',
        'disagree': 'No concuerdan',
        'vs': 'vs',
        'probability_distances_pdf': '📏 Distancias entre Distribuciones de Probabilidad',
        'clinical_interpretation_pdf': '🏥 Interpretación Clínica',
        'consensus_classification_pdf': 'Clasificación consenso',
        'average_confidence_pdf': 'Confianza promedio',
        'description_pdf': 'Descripción',
        'important_note': '⚠️ Nota Importante',
        
        # Instructions
        'instructions_title': '👆 Instrucciones',
        'instructions_intro': 'Para comenzar el análisis, por favor:',
        'instruction_1': '📤 Carga una imagen histopatológica usando el selector de archivos arriba',
        'instruction_2': '🚀 Presiona "Iniciar Análisis" para procesar la imagen con los 4 modelos de IA',
        'instruction_3': '📊 Revisa los resultados detallados y las métricas estadísticas',
        'instruction_4': '📄 Descarga el reporte PDF con todos los hallazgos',
        
        # System capabilities
        'system_capabilities': '🔬 Capacidades del Sistema',
        'system_capabilities_desc': 'Este sistema puede analizar imágenes histopatológicas de osteosarcoma utilizando 4 modelos de deep learning diferentes:',
        'vgg16_full_desc': 'Arquitectura clásica profunda para análisis detallado',
        'resnet50_full_desc': 'Red residual con conexiones skip para mejor precisión',
        'mobilenetv2_full_desc': 'Modelo eficiente optimizado para velocidad',
        'efficientnetb0_full_desc': 'Balance óptimo entre precisión y eficiencia',
        
        # Classification classes info
        'classification_classes': '🎯 Clases de Clasificación',
        
        # Medical disclaimer
        'medical_disclaimer': '⚠️ Importante - Disclaimer Médico',
        'disclaimer_text': 'Este sistema es una herramienta de apoyo diagnóstico basada en inteligencia artificial. Los resultados presentados NO constituyen un diagnóstico médico definitivo y deben ser interpretados exclusivamente por profesionales médicos cualificados.',
        'disclaimer_recommendations': 'Se recomienda encarecidamente:',
        'disclaimer_item_1': 'Correlacionar estos resultados con estudios clínicos, radiológicos e histopatológicos adicionales',
        'disclaimer_item_2': 'Considerar el contexto clínico completo del paciente',
        'disclaimer_item_3': 'Seguir las guidelines y protocolos médicos establecidos',
        'disclaimer_item_4': 'Buscar opinión de especialistas en oncología y patología cuando sea apropiado',
        'disclaimer_final': 'Este análisis no sustituye el juicio clínico profesional.',
        
        # PDF disclaimer
        'pdf_disclaimer_title': 'Este análisis es una herramienta de apoyo diagnóstico basada en inteligencia artificial.',
        'pdf_disclaimer_main': 'Los resultados deben ser interpretados por un profesional médico cualificado y NO sustituyen el juicio clínico profesional.',
        'pdf_disclaimer_recommendations': 'Se recomienda encarecidamente:',
        'pdf_disclaimer_item_1': 'Correlacionar estos resultados con otros estudios clínicos, radiológicos e histopatológicos',
        'pdf_disclaimer_item_2': 'Considerar el contexto clínico completo del paciente',
        'pdf_disclaimer_item_3': 'Seguir las guidelines y protocolos médicos establecidos',
        'pdf_disclaimer_item_4': 'Buscar opinión de especialistas en oncología y patología cuando sea apropiado',
        'pdf_disclaimer_final': 'Este análisis no sustituye el juicio clínico profesional.',
        
        # Charts and labels
        'tissue_type': 'Tipo de Tejido',
        'comparison': 'Comparación',
        'number_of_comparisons': 'Número de Comparaciones',
        'tissue_types': 'Tipos de Tejido',
        'models': 'Modelos',
        'color': 'color',
        
        # Time and date
        'date_format': '%d/%m/%Y %H:%M',
        
        # Common words
        'yes': 'Sí',
        'no': 'No',
        'and': 'y',
        'or': 'o',
        'with': 'con',
        'without': 'sin',
        'for': 'para',
        'from': 'de',
        'to': 'a',
        'in': 'en',
        'of': 'de',
        'the': 'el/la',
        'a': 'un/una',
        'an': 'un/una',
        'kb': 'KB',
        'px': 'px',
        'models_count': 'modelos',
        'agree_text': 'Concuerdan',
        'disagree_text': 'No concuerdan',
    },
    
    'en': {
        # Page configuration
        'page_title': 'Osteosarcoma Classifier',
        'page_icon': '🔬',
        
        # Main headers
        'main_header': '🔬 Osteosarcoma Classifier',
        'sub_header_upload': '📤 Upload Image for Analysis',
        'sub_header_results': '📊 Analysis Results',
        
        # Sidebar
        'sidebar_system_info': '📋 System Information',
        'sidebar_objective': '🎯 Objective',
        'sidebar_objective_desc': 'Automatic classification of histopathological images for osteosarcoma detection and analysis using 4 deep learning models.',
        'sidebar_available_models': '🤖 Available Models',
        'sidebar_diagnosis_classes': '🏥 Diagnosis Classes',
        'sidebar_language': '🌐 Language',
        
        # Model descriptions
        'model_vgg16_desc': 'Deep classical architecture',
        'model_resnet50_desc': 'Advanced residual network',
        'model_mobilenetv2_desc': 'Efficient and lightweight model',
        'model_efficientnetb0_desc': 'Optimal precision/efficiency balance',
        
        # Class descriptions
        'class_non_tumor': 'Tissue without tumor presence',
        'class_non_viable_tumor': 'Non-viable tumor tissue (necrotic)',
        'class_viable': 'Viable tumor tissue (active)',
        'class_mixed': 'Mixed tissue with combined characteristics',
        
        # File upload
        'file_upload_label': 'Select a histopathological image',
        'file_upload_help': 'Supported formats: PNG, JPG, JPEG, TIFF, BMP',
        'supported_formats': 'Supported formats',
        
        # Image info
        'image_info_title': 'Image information:',
        'image_dimensions': 'Dimensions',
        'image_mode': 'Mode',
        'image_format': 'Format',
        'image_size': 'Size',
        'image_original': 'Original Image',
        'image_resized': 'Resized Image (224×224)',
        'image_resize_note': 'The image will be resized to 224×224 pixels for analysis, maintaining the necessary quality for classification.',
        
        # Buttons
        'btn_start_analysis': '🚀 Start Analysis',
        'btn_new_analysis': '🔄 New Analysis',
        'btn_generate_pdf': '📥 Generate PDF Report',
        'btn_download_pdf': '⬇️ Download PDF Report',
        
        # Loading messages
        'loading_models': '🔄 Loading AI models...',
        'loading_analysis': '🔍 Analyzing image with AI models...',
        'loading_pdf': '📝 Generating PDF report...',
        
        # Success messages
        'models_loaded': 'models loaded successfully',
        'pdf_generated': '✅ PDF report generated successfully',
        'file_size': 'File size',
        
        # Error messages
        'error_loading_models': '❌ Could not load models. Check that .h5 files are in the \'models/\' folder',
        'error_loading_model': 'Error loading model',
        'error_prediction': 'Error in prediction with',
        'error_inserting_image': 'Error inserting image',
        'error_no_valid_predictions': '❌ No valid predictions to generate report',
        'error_generating_pdf': '❌ Error generating PDF',
        'error_empty_pdf': '❌ Error: PDF file is empty',
        'error_pdf_buffer': '❌ Error generating PDF buffer',
        'error_reload_suggestion': '💡 Try reloading the page and uploading the image again',
        'error_show_details': '🔍 Show error details',
        
        # Analysis results
        'consensus_classification': '🎯 Consensus Classification',
        'average_confidence': '📈 Average Confidence',
        'total_agreement': '🤝 Total Agreement',
        'agreement_yes': '✅ Yes',
        'agreement_no': '⚠️ No',
        'detailed_results': '🔍 Detailed Results by Model',
        'prediction': 'Prediction',
        'confidence': 'Confidence',
        'description': 'Description',
        'class': 'Class',
        'probability': 'Probability',
        
        # Visualizations
        'visualizations': '📈 Visualizations',
        'probability_distribution': 'Probability Distribution by Model',
        'probability_heatmap': 'Heat Map - Probabilities by Model',
        'model_agreement': 'Model Agreement',
        'probability_distances': 'Probability Distribution Distances',
        
        # Statistical analysis
        'statistical_analysis': '📊 Detailed Statistical Metrics',
        'agreement_analysis': '🤝 Agreement Analysis between Models',
        'distance_analysis': '📏 Probability Distribution Distances',
        'agreement_concordance': '🔗 Agreement and Distance Analysis',
        
        # Statistical table headers
        'model_1': 'Model 1',
        'model_2': 'Model 2',
        'prediction_1': 'Prediction 1',
        'prediction_2': 'Prediction 2',
        'concordance': 'Concordance',
        'euclidean_distance': 'Euclidean Distance',
        'cosine_distance': 'Cosine Distance',
        'kl_divergence': 'KL Divergence',
        
        # Distance interpretation
        'distance_interpretation': '💡 Distance Interpretation:',
        'euclidean_desc': 'Geometric distance between probability vectors (lower = higher similarity)',
        'cosine_desc': 'Angular similarity between vectors (lower = higher similarity)',
        'kl_desc': 'Measure of difference between distributions (lower = higher similarity)',
        
        # Clinical interpretation
        'clinical_interpretation': '🏥 Clinical Interpretation',
        'diagnostic_summary': '📋 Diagnostic Summary',
        'main_classification': 'Main classification',
        'consensus_confidence': 'Consensus confidence',
        'model_concordance': 'Model concordance',
        'concordance_high': 'High',
        'concordance_partial': 'Partial',
        'clinical_recommendation': '💊 Clinical Recommendation',
        
        # Clinical recommendations
        'recommendation_non_tumor': '👍 Analysis suggests absence of tumor tissue. Routine follow-up according to clinical protocol is recommended.',
        'recommendation_non_viable': '⚡ Non-viable tumor tissue (necrotic) detected. Evaluate response to previous treatment and consider therapeutic adjustments.',
        'recommendation_viable': '⚠️ Viable tumor tissue detected. Consider immediate treatment options according to oncological guidelines.',
        'recommendation_mixed': '🔄 Mixed tissue with heterogeneous characteristics detected. Additional histopathological analysis and multidisciplinary evaluation recommended.',
        
        # PDF Report
        'export_report': '📄 Export Report',
        'pdf_report_title': '🔬 Osteosarcoma Analysis Report',
        'analysis_info': '📋 Analysis Information',
        'analysis_date': 'Analysis date',
        'models_used': 'Models used',
        'classes_analyzed': 'Classes analyzed',
        'analyzed_image': '🖼️ Analyzed Image',
        'prediction_results': '🤖 Prediction Results',
        'confidence_label': 'Confidence',
        'probability_distribution_label': 'Probability distribution',
        'agreement_analysis_pdf': '🤝 Agreement Analysis',
        'all_models_agree': '✅ All models agree on the prediction',
        'models_disagree': '⚠️ Models do not completely agree',
        'pair_details': 'Details by model pairs',
        'agree': 'Agree',
        'disagree': 'Disagree',
        'vs': 'vs',
        'probability_distances_pdf': '📏 Probability Distribution Distances',
        'clinical_interpretation_pdf': '🏥 Clinical Interpretation',
        'consensus_classification_pdf': 'Consensus classification',
        'average_confidence_pdf': 'Average confidence',
        'description_pdf': 'Description',
        'important_note': '⚠️ Important Note',
        
        # Instructions
        'instructions_title': '👆 Instructions',
        'instructions_intro': 'To start the analysis, please:',
        'instruction_1': '📤 Upload a histopathological image using the file selector above',
        'instruction_2': '🚀 Press "Start Analysis" to process the image with the 4 AI models',
        'instruction_3': '📊 Review the detailed results and statistical metrics',
        'instruction_4': '📄 Download the PDF report with all findings',
        
        # System capabilities
        'system_capabilities': '🔬 System Capabilities',
        'system_capabilities_desc': 'This system can analyze histopathological images of osteosarcoma using 4 different deep learning models:',
        'vgg16_full_desc': 'Deep classical architecture for detailed analysis',
        'resnet50_full_desc': 'Residual network with skip connections for better precision',
        'mobilenetv2_full_desc': 'Efficient model optimized for speed',
        'efficientnetb0_full_desc': 'Optimal balance between precision and efficiency',
        
        # Classification classes info
        'classification_classes': '🎯 Classification Classes',
        
        # Medical disclaimer
        'medical_disclaimer': '⚠️ Important - Medical Disclaimer',
        'disclaimer_text': 'This system is a diagnostic support tool based on artificial intelligence. The results presented do NOT constitute a definitive medical diagnosis and must be interpreted exclusively by qualified medical professionals.',
        'disclaimer_recommendations': 'It is strongly recommended to:',
        'disclaimer_item_1': 'Correlate these results with additional clinical, radiological and histopathological studies',
        'disclaimer_item_2': 'Consider the complete clinical context of the patient',
        'disclaimer_item_3': 'Follow established medical guidelines and protocols',
        'disclaimer_item_4': 'Seek opinion from oncology and pathology specialists when appropriate',
        'disclaimer_final': 'This analysis does not substitute professional clinical judgment.',
        
        # PDF disclaimer
        'pdf_disclaimer_title': 'This analysis is a diagnostic support tool based on artificial intelligence.',
        'pdf_disclaimer_main': 'Results must be interpreted by a qualified medical professional and do NOT substitute professional clinical judgment.',
        'pdf_disclaimer_recommendations': 'It is strongly recommended to:',
        'pdf_disclaimer_item_1': 'Correlate these results with other clinical, radiological and histopathological studies',
        'pdf_disclaimer_item_2': 'Consider the complete clinical context of the patient',
        'pdf_disclaimer_item_3': 'Follow established medical guidelines and protocols',
        'pdf_disclaimer_item_4': 'Seek opinion from oncology and pathology specialists when appropriate',
        'pdf_disclaimer_final': 'This analysis does not substitute professional clinical judgment.',
        
        # Charts and labels
        'tissue_type': 'Tissue Type',
        'comparison': 'Comparison',
        'number_of_comparisons': 'Number of Comparisons',
        'tissue_types': 'Tissue Types',
        'models': 'Models',
        'color': 'color',
        
        # Time and date
        'date_format': '%m/%d/%Y %H:%M',
        
        # Common words
        'yes': 'Yes',
        'no': 'No',
        'and': 'and',
        'or': 'or',
        'with': 'with',
        'without': 'without',
        'for': 'for',
        'from': 'from',
        'to': 'to',
        'in': 'in',
        'of': 'of',
        'the': 'the',
        'a': 'a',
        'an': 'an',
        'kb': 'KB',
        'px': 'px',
        'models_count': 'models',
        'agree_text': 'Agree',
        'disagree_text': 'Disagree',
    }
}

# Class names and descriptions for each language
CLASS_NAMES = {
    'es': ['Sin-Tumor', 'Tumor-No-Viable', 'Viable', 'Mixto'],
    'en': ['Non-Tumor', 'Non-Viable-Tumor', 'Viable', 'Mixed']
}

CLASS_DESCRIPTIONS = {
    'es': {
        'Sin-Tumor': 'Tejido sin presencia de tumor',
        'Tumor-No-Viable': 'Tejido tumoral no viable (necrótico)',
        'Viable': 'Tejido tumoral viable (activo)',
        'Mixto': 'Tejido mixto con características combinadas'
    },
    'en': {
        'Non-Tumor': 'Tissue without tumor presence',
        'Non-Viable-Tumor': 'Non-viable tumor tissue (necrotic)',
        'Viable': 'Viable tumor tissue (active)',
        'Mixed': 'Mixed tissue with combined characteristics'
    }
}

# Model descriptions for each language
MODEL_DESCRIPTIONS = {
    'es': {
        'VGG16': 'Arquitectura clásica con capas convolucionales profundas',
        'ResNet50': 'Red residual con conexiones skip para mejor entrenamiento',
        'MobileNetV2': 'Arquitectura eficiente para dispositivos móviles',
        'EfficientNetB0': 'Modelo balanceado entre precisión y eficiencia'
    },
    'en': {
        'VGG16': 'Classical architecture with deep convolutional layers',
        'ResNet50': 'Residual network with skip connections for better training',
        'MobileNetV2': 'Efficient architecture for mobile devices',
        'EfficientNetB0': 'Balanced model between precision and efficiency'
    }
}

def get_text(key, language='es'):
    """
    Get translated text for a given key and language
    
    Args:
        key (str): The key to look up in the translations
        language (str): Language code ('es' or 'en')
    
    Returns:
        str: Translated text or the key itself if not found
    """
    try:
        return TRANSLATIONS[language][key]
    except KeyError:
        # Fallback to Spanish if key not found in requested language
        try:
            return TRANSLATIONS['es'][key]
        except KeyError:
            # Return the key itself if not found anywhere
            return key

def get_class_names(language='es'):
    """Get class names for the specified language"""
    return CLASS_NAMES.get(language, CLASS_NAMES['es'])

def get_class_descriptions(language='es'):
    """Get class descriptions for the specified language"""
    return CLASS_DESCRIPTIONS.get(language, CLASS_DESCRIPTIONS['es'])

def get_model_descriptions(language='es'):
    """Get model descriptions for the specified language"""
    return MODEL_DESCRIPTIONS.get(language, MODEL_DESCRIPTIONS['es'])

def get_available_languages():
    """Get list of available languages"""
    return list(TRANSLATIONS.keys())

def get_language_names():
    """Get human-readable language names"""
    return {
        'es': 'Español',
        'en': 'English'
    }
