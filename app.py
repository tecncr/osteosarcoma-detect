import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from scipy.spatial.distance import euclidean, cosine
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import tempfile
import os
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Clasificador de Osteosarcoma",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para diseño responsivo y moderno
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        text-align: center;
        color: #2E86AB;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        font-weight: bold;
        margin: 1.5rem 0;
    }
    
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-left: 5px solid #2E86AB;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-left: 5px solid #ffc107;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-left: 5px solid #28a745;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Función para cargar modelos con caché
@st.cache_resource
def load_models():
    """Carga todos los modelos pre-entrenados"""
    models = {}
    model_info = {
        'VGG16': {
            'path': 'models/VGG16_osteosarcoma.h5',
            'preprocess': vgg_preprocess,
            'description': 'Arquitectura clásica con capas convolucionales profundas'
        },
        'ResNet50': {
            'path': 'models/ResNet50_osteosarcoma.h5',
            'preprocess': resnet_preprocess,
            'description': 'Red residual con conexiones skip para mejor entrenamiento'
        },
        'MobileNetV2': {
            'path': 'models/MobileNetV2_osteosarcoma.h5',
            'preprocess': mobilenet_preprocess,
            'description': 'Arquitectura eficiente para dispositivos móviles'
        },
        'EfficientNetB0': {
            'path': 'models/EfficientNetB0_osteosarcoma.h5',
            'preprocess': efficientnet_preprocess,
            'description': 'Modelo balanceado entre precisión y eficiencia'
        }
    }
    
    for model_name, info in model_info.items():
        try:
            models[model_name] = {
                'model': load_model(info['path']),
                'preprocess': info['preprocess'],
                'description': info['description']
            }
        except Exception as e:
            st.error(f"Error cargando modelo {model_name}: {str(e)}")
    
    return models

# Configuración de clases
CLASS_NAMES = ['Non-Tumor', 'Non-Viable-Tumor', 'Viable', 'Mixed']
CLASS_DESCRIPTIONS = {
    'Non-Tumor': 'Tejido sin presencia de tumor',
    'Non-Viable-Tumor': 'Tejido tumoral no viable (necrótico)',
    'Viable': 'Tejido tumoral viable (activo)',
    'Mixed': 'Tejido mixto con características combinadas'
}

def preprocess_image(image, target_size=(224, 224)):
    """Preprocesa una imagen para inferencia"""
    # Redimensionar imagen
    img = image.resize(target_size)
    img_array = np.array(img)
    
    # Asegurar 3 canales (RGB)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array, img_array, img_array], axis=-1)
    elif img_array.shape[2] == 1:
        img_array = np.concatenate([img_array, img_array, img_array], axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    
    return img_array

def predict_with_models(image, models):
    """Realiza inferencia con todos los modelos"""
    predictions = {}
    probabilities = {}
    
    # Preprocesar imagen base
    img_array = preprocess_image(image)
    
    for model_name, model_info in models.items():
        try:
            # Aplicar preprocesamiento específico del modelo
            img_processed = model_info['preprocess'](np.expand_dims(img_array, axis=0))
            
            # Realizar predicción
            pred_probs = model_info['model'].predict(img_processed, verbose=0)
            pred_class = np.argmax(pred_probs, axis=1)[0]
            
            predictions[model_name] = pred_class
            probabilities[model_name] = pred_probs[0]
            
        except Exception as e:
            st.error(f"Error en predicción con {model_name}: {str(e)}")
            predictions[model_name] = None
            probabilities[model_name] = None
    
    return predictions, probabilities

def calculate_agreement_metrics(predictions):
    """Calcula métricas de concordancia entre modelos"""
    # Filtrar predicciones válidas
    valid_preds = {k: v for k, v in predictions.items() if v is not None}
    
    if len(valid_preds) < 2:
        return {}
    
    metrics = {}
    model_names = list(valid_preds.keys())
    
    # Calcular concordancia por pares
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            model1, model2 = model_names[i], model_names[j]
            
            # Concordancia simple
            agreement = 1 if valid_preds[model1] == valid_preds[model2] else 0
            
            metrics[f"{model1}_vs_{model2}"] = {
                'agreement': agreement,
                'pred1': valid_preds[model1],
                'pred2': valid_preds[model2]
            }
    
    # Concordancia general
    all_same = len(set(valid_preds.values())) == 1
    metrics['all_models_agree'] = all_same
    
    return metrics

def calculate_probability_distances(probabilities):
    """Calcula distancias entre distribuciones de probabilidad"""
    distances = {}
    model_names = list(probabilities.keys())
    
    # Filtrar probabilidades válidas
    valid_probs = {k: v for k, v in probabilities.items() if v is not None}
    
    if len(valid_probs) < 2:
        return distances
    
    model_names = list(valid_probs.keys())
    
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            model1, model2 = model_names[i], model_names[j]
            
            prob1 = valid_probs[model1]
            prob2 = valid_probs[model2]
            
            # Distancia euclidiana
            euclidean_dist = euclidean(prob1, prob2)
            
            # Distancia coseno
            cosine_dist = cosine(prob1, prob2)
            
            # Divergencia KL (aproximada)
            epsilon = 1e-10
            prob1_safe = prob1 + epsilon
            prob2_safe = prob2 + epsilon
            kl_div = np.sum(prob1_safe * np.log(prob1_safe / prob2_safe))
            
            distances[f"{model1}_vs_{model2}"] = {
                'euclidean': euclidean_dist,
                'cosine': cosine_dist,
                'kl_divergence': kl_div
            }
    
    return distances

def create_prediction_visualization(probabilities, predictions):
    """Crea visualizaciones de las predicciones"""
    
    # Preparar datos para gráficos
    valid_probs = {k: v for k, v in probabilities.items() if v is not None}
    
    if not valid_probs:
        return None, None
    
    # Crear DataFrame para facilitar la visualización
    prob_data = []
    for model_name, probs in valid_probs.items():
        for i, class_name in enumerate(CLASS_NAMES):
            prob_data.append({
                'Modelo': model_name,
                'Clase': class_name,
                'Probabilidad': probs[i],
                'Predicción': predictions[model_name] == i
            })
    
    df_probs = pd.DataFrame(prob_data)
    
    # Gráfico de barras comparativo
    fig1 = px.bar(
        df_probs, 
        x='Clase', 
        y='Probabilidad', 
        color='Modelo',
        title='Distribución de Probabilidades por Modelo',
        labels={'Probabilidad': 'Probabilidad (%)', 'Clase': 'Tipo de Tejido'},
        template='plotly_white',
        height=500
    )
    
    # Formatear probabilidades como porcentaje
    fig1.update_traces(texttemplate='%{y:.1%}', textposition='auto')
    fig1.update_layout(
        title_x=0.5,
        font=dict(size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Heatmap de probabilidades
    prob_matrix = df_probs.pivot(index='Modelo', columns='Clase', values='Probabilidad')
    
    fig2 = px.imshow(
        prob_matrix.values,
        x=prob_matrix.columns,
        y=prob_matrix.index,
        title='Mapa de Calor - Probabilidades por Modelo',
        aspect='auto',
        color_continuous_scale='viridis',
        labels={'color': 'Probabilidad'},
        height=400
    )
    
    # Añadir anotaciones con los valores
    annotations = []
    for i, modelo in enumerate(prob_matrix.index):
        for j, clase in enumerate(prob_matrix.columns):
            annotations.append(
                dict(
                    x=j, y=i,
                    text=f"{prob_matrix.iloc[i, j]:.1%}",
                    showarrow=False,
                    font=dict(color="white" if prob_matrix.iloc[i, j] < 0.5 else "black")
                )
            )
    
    fig2.update_layout(annotations=annotations, title_x=0.5)
    
    return fig1, fig2

def create_agreement_visualization(agreement_metrics, probability_distances):
    """Crea visualizaciones de concordancia y distancias"""
    
    if not agreement_metrics or not probability_distances:
        return None, None
    
    # Preparar datos de concordancia
    agreement_data = []
    distance_data = []
    
    for comparison, metrics in agreement_metrics.items():
        if comparison != 'all_models_agree':
            models = comparison.split('_vs_')
            agreement_data.append({
                'Comparación': f"{models[0]} vs {models[1]}",
                'Concordancia': 'Sí' if metrics['agreement'] else 'No',
                'Predicción 1': CLASS_NAMES[metrics['pred1']],
                'Predicción 2': CLASS_NAMES[metrics['pred2']]
            })
    
    # Preparar datos de distancias
    for comparison, distances in probability_distances.items():
        models = comparison.split('_vs_')
        distance_data.append({
            'Comparación': f"{models[0]} vs {models[1]}",
            'Distancia Euclidiana': distances['euclidean'],
            'Distancia Coseno': distances['cosine'],
            'Divergencia KL': distances['kl_divergence']
        })
    
    # Gráfico de concordancia
    df_agreement = pd.DataFrame(agreement_data)
    
    fig1 = px.bar(
        df_agreement,
        x='Comparación',
        color='Concordancia',
        title='Concordancia entre Modelos',
        labels={'count': 'Número de Comparaciones'},
        template='plotly_white',
        height=400
    )
    
    fig1.update_layout(title_x=0.5, xaxis_tickangle=-45)
    
    # Gráfico de distancias
    df_distances = pd.DataFrame(distance_data)
    
    fig2 = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Distancia Euclidiana', 'Distancia Coseno', 'Divergencia KL'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Añadir barras para cada métrica de distancia
    for i, metric in enumerate(['Distancia Euclidiana', 'Distancia Coseno', 'Divergencia KL']):
        fig2.add_trace(
            go.Bar(
                x=df_distances['Comparación'],
                y=df_distances[metric],
                name=metric,
                showlegend=False
            ),
            row=1, col=i+1
        )
    
    fig2.update_layout(
        title_text="Distancias entre Distribuciones de Probabilidad",
        title_x=0.5,
        height=500
    )
    
    # Rotar etiquetas del eje x
    fig2.update_xaxes(tickangle=-45)
    
    return fig1, fig2

def generate_pdf_report(image, predictions, probabilities, agreement_metrics, probability_distances):
    """Genera un reporte en PDF con todos los resultados"""
    
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=72, bottomMargin=72)
        styles = getSampleStyleSheet()
        story = []
        
        # Estilos personalizados
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=20,
            spaceAfter=30,
            alignment=1,  # Centrado
            textColor=colors.darkblue,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=15,
            spaceBefore=20,
            textColor=colors.darkred,
            fontName='Helvetica-Bold'
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            fontName='Helvetica'
        )
        
        # Título del reporte
        story.append(Paragraph("🔬 Reporte de Análisis de Osteosarcoma", title_style))
        story.append(Spacer(1, 30))
        
        # Información general
        story.append(Paragraph("📋 Información del Análisis", heading_style))
        story.append(Paragraph(f"<b>Fecha de análisis:</b> {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}", normal_style))
        story.append(Paragraph(f"<b>Modelos utilizados:</b> VGG16, ResNet50, MobileNetV2, EfficientNetB0", normal_style))
        story.append(Paragraph(f"<b>Clases analizadas:</b> {', '.join(CLASS_NAMES)}", normal_style))
        story.append(Spacer(1, 20))
        
        # Crear imagen temporal y añadirla al reporte
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png', dir=tempfile.gettempdir()) as tmp_file:
            # Asegurar que la imagen está en formato RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Redimensionar imagen para el PDF manteniendo aspect ratio
            max_size = (300, 300)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            image.save(tmp_file.name, 'PNG', quality=95)
            
            # Añadir imagen al reporte
            story.append(Paragraph("🖼️ Imagen Analizada", heading_style))
            
            try:
                img_width, img_height = image.size
                # Calcular dimensiones para el PDF manteniendo aspect ratio
                max_width = 4 * inch
                max_height = 3 * inch
                
                if img_width > img_height:
                    pdf_width = max_width
                    pdf_height = (img_height / img_width) * max_width
                else:
                    pdf_height = max_height
                    pdf_width = (img_width / img_height) * max_height
                
                img_for_pdf = RLImage(tmp_file.name, width=pdf_width, height=pdf_height)
                story.append(img_for_pdf)
                story.append(Spacer(1, 20))
                
            except Exception as img_error:
                story.append(Paragraph(f"<i>Error al insertar imagen: {str(img_error)}</i>", normal_style))
                story.append(Spacer(1, 20))
        
        # Resultados de predicción
        story.append(Paragraph("🤖 Resultados de Predicción", heading_style))
        
        valid_preds = {k: v for k, v in predictions.items() if v is not None}
        valid_probs = {k: v for k, v in probabilities.items() if v is not None}
        
        if valid_preds:
            for model_name in valid_preds.keys():
                pred_class = valid_preds[model_name]
                pred_probs = valid_probs[model_name]
                
                story.append(Paragraph(f"<b>{model_name}:</b>", normal_style))
                story.append(Paragraph(f"• Predicción: <b>{CLASS_NAMES[pred_class]}</b>", normal_style))
                story.append(Paragraph(f"• Confianza: <b>{pred_probs[pred_class]:.1%}</b>", normal_style))
                
                # Añadir todas las probabilidades
                prob_text = "• Distribución de probabilidades:<br/>"
                for i, class_name in enumerate(CLASS_NAMES):
                    prob_text += f"&nbsp;&nbsp;&nbsp;&nbsp;- {class_name}: {pred_probs[i]:.1%}<br/>"
                
                story.append(Paragraph(prob_text, normal_style))
                story.append(Spacer(1, 10))
        
        # Análisis de concordancia
        story.append(Paragraph("🤝 Análisis de Concordancia", heading_style))
        
        if agreement_metrics:
            if agreement_metrics.get('all_models_agree', False):
                story.append(Paragraph("✅ <b>Todos los modelos están de acuerdo</b> en la predicción", normal_style))
            else:
                story.append(Paragraph("⚠️ <b>Los modelos no concuerdan completamente</b>", normal_style))
            
            # Detalles de concordancia por pares
            story.append(Paragraph("<b>Detalles por pares de modelos:</b>", normal_style))
            for comparison, metrics in agreement_metrics.items():
                if comparison != 'all_models_agree':
                    models = comparison.split('_vs_')
                    agreement_text = "Concuerdan" if metrics['agreement'] else "No concuerdan"
                    story.append(Paragraph(
                        f"• {models[0]} vs {models[1]}: <b>{agreement_text}</b> "
                        f"({CLASS_NAMES[metrics['pred1']]} vs {CLASS_NAMES[metrics['pred2']]})",
                        normal_style
                    ))
        
        story.append(Spacer(1, 20))
        
        # Distancias entre modelos
        if probability_distances:
            story.append(Paragraph("📏 Distancias entre Distribuciones de Probabilidad", heading_style))
            
            for comparison, distances in probability_distances.items():
                models = comparison.split('_vs_')
                story.append(Paragraph(f"<b>{models[0]} vs {models[1]}:</b>", normal_style))
                story.append(Paragraph(f"• Distancia Euclidiana: {distances['euclidean']:.4f}", normal_style))
                story.append(Paragraph(f"• Distancia Coseno: {distances['cosine']:.4f}", normal_style))
                story.append(Paragraph(f"• Divergencia KL: {distances['kl_divergence']:.4f}", normal_style))
                story.append(Spacer(1, 10))
        
        # Interpretación médica
        story.append(Paragraph("🏥 Interpretación Clínica", heading_style))
        
        # Determinar la clase más probable
        if valid_probs:
            avg_probs = np.mean([probs for probs in valid_probs.values()], axis=0)
            most_likely_class = np.argmax(avg_probs)
            confidence = avg_probs[most_likely_class]
            
            story.append(Paragraph(f"<b>Clasificación consenso:</b> {CLASS_NAMES[most_likely_class]}", normal_style))
            story.append(Paragraph(f"<b>Confianza promedio:</b> {confidence:.1%}", normal_style))
            story.append(Paragraph(f"<b>Descripción:</b> {CLASS_DESCRIPTIONS[CLASS_NAMES[most_likely_class]]}", normal_style))
            
            # Recomendaciones basadas en el resultado
            recommendations = {
                0: "El análisis sugiere ausencia de tejido tumoral. Se recomienda seguimiento rutinario según protocolo clínico.",
                1: "Se detecta tejido tumoral no viable (necrótico). Evaluar respuesta al tratamiento previo y considerar ajustes terapéuticos.",
                2: "Se detecta tejido tumoral viable activo. Considerar opciones de tratamiento inmediato según guidelines oncológicos.",
                3: "Se detecta tejido mixto con características heterogéneas. Se recomienda análisis histopatológico adicional y evaluación multidisciplinaria."
            }
            
            recommendation = recommendations.get(most_likely_class, "Consultar con especialista para interpretación adicional.")
            
            story.append(Spacer(1, 15))
            story.append(Paragraph(f"<b>💊 Recomendación Clínica:</b>", normal_style))
            story.append(Paragraph(recommendation, normal_style))
        
        # Disclaimer
        story.append(Spacer(1, 30))
        story.append(Paragraph("⚠️ Nota Importante", heading_style))
        
        disclaimer_text = """
        <b>Este análisis es una herramienta de apoyo diagnóstico basada en inteligencia artificial.</b><br/><br/>
        
        Los resultados deben ser interpretados por un profesional médico cualificado y <b>NO sustituyen 
        el juicio clínico profesional</b>.<br/><br/>
        
        <b>Se recomienda encarecidamente:</b><br/>
        • Correlacionar estos resultados con otros estudios clínicos, radiológicos e histopatológicos<br/>
        • Considerar el contexto clínico completo del paciente<br/>
        • Seguir las guidelines y protocolos médicos establecidos<br/>
        • Buscar opinión de especialistas en oncología y patología cuando sea apropiado<br/><br/>
        
        <b>Este análisis no sustituye el juicio clínico profesional.</b>
        """
        
        story.append(Paragraph(disclaimer_text, normal_style))
        
        # Construir PDF
        doc.build(story)
        buffer.seek(0)
        
        # Limpiar archivo temporal
        try:
            os.unlink(tmp_file.name)
        except:
            pass
        
        return buffer
        
    except Exception as e:
        # En caso de error, crear un PDF simple con el error
        error_buffer = BytesIO()
        error_doc = SimpleDocTemplate(error_buffer, pagesize=letter)
        error_story = []
        
        error_story.append(Paragraph("Error en Generación de PDF", styles['Title']))
        error_story.append(Spacer(1, 20))
        error_story.append(Paragraph(f"Se produjo un error al generar el reporte: {str(e)}", styles['Normal']))
        error_story.append(Spacer(1, 20))
        error_story.append(Paragraph("Por favor, intente nuevamente o contacte al soporte técnico.", styles['Normal']))
        
        error_doc.build(error_story)
        error_buffer.seek(0)
        return error_buffer

def main():
    # Título principal
    st.markdown('<h1 class="main-header">🔬 Clasificador de Osteosarcoma</h1>', unsafe_allow_html=True)
    
    # Información en el sidebar
    with st.sidebar:
        st.markdown("### 📋 Información del Sistema")
        
        st.markdown("""
        <div class="info-box">
        <h4>🎯 Objetivo</h4>
        <p>Clasificación automática de imágenes histopatológicas para detección y análisis de osteosarcoma utilizando 4 modelos de deep learning.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🤖 Modelos Disponibles")
        
        model_descriptions = {
            'VGG16': 'Arquitectura clásica profunda',
            'ResNet50': 'Red residual avanzada',
            'MobileNetV2': 'Modelo eficiente y ligero',
            'EfficientNetB0': 'Balance óptimo precisión/eficiencia'
        }
        
        for model, desc in model_descriptions.items():
            st.markdown(f"""
            <div class="model-card">
            <strong>{model}</strong><br>
            <small>{desc}</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 🏥 Clases de Diagnóstico")
        
        for class_name, description in CLASS_DESCRIPTIONS.items():
            st.markdown(f"**{class_name}:** {description}")
    
    # Cargar modelos
    with st.spinner('🔄 Cargando modelos de IA...'):
        models = load_models()
    
    if not models:
        st.error("❌ No se pudieron cargar los modelos. Verifica que los archivos .h5 estén en la carpeta 'models/'")
        return
    
    st.success(f"✅ {len(models)} modelos cargados exitosamente")
    
    # Sección de carga de imagen
    st.markdown('<h2 class="sub-header">📤 Cargar Imagen para Análisis</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Selecciona una imagen histopatológica",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        help="Formatos soportados: PNG, JPG, JPEG, TIFF, BMP"
    )
    
    if uploaded_file is not None:
        # Guardar la imagen cargada en session_state
        if 'uploaded_image' not in st.session_state or st.session_state.get('uploaded_filename') != uploaded_file.name:
            st.session_state.uploaded_image = Image.open(uploaded_file)
            st.session_state.uploaded_filename = uploaded_file.name
            # Limpiar análisis previo cuando se carga nueva imagen
            for key in ['predictions', 'probabilities', 'agreement_metrics', 
                       'probability_distances', 'analyzed_image', 'analysis_completed']:
                if key in st.session_state:
                    del st.session_state[key]
        
        image = st.session_state.uploaded_image
        
        # Mostrar imagen cargada
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption="Imagen Original", use_container_width=True)
            
            # Información de la imagen
            st.markdown(f"""
            <div class="info-box">
            <strong>Información de la imagen:</strong><br>
            📏 Dimensiones: {image.size[0]} × {image.size[1]} px<br>
            🎨 Modo: {image.mode}<br>
            📁 Formato: {image.format}<br>
            💾 Tamaño: {len(uploaded_file.getvalue()) / 1024:.1f} KB
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Imagen redimensionada para análisis
            resized_image = image.resize((224, 224))
            st.image(resized_image, caption="Imagen Redimensionada (224×224)", use_container_width=True)
            
            st.markdown("""
            <div class="warning-box">
            <strong>⚠️ Nota:</strong> La imagen será redimensionada a 224×224 píxeles para el análisis, 
            manteniendo la calidad necesaria para la clasificación.
            </div>
            """, unsafe_allow_html=True)
        
        # Botón de análisis
        if st.button("🚀 Iniciar Análisis", type="primary", use_container_width=True):
            with st.spinner('🔍 Analizando imagen con modelos de IA...'):
                # Realizar predicciones
                predictions, probabilities = predict_with_models(image, models)
                
                # Calcular métricas de concordancia
                agreement_metrics = calculate_agreement_metrics(predictions)
                probability_distances = calculate_probability_distances(probabilities)
                
                # Guardar resultados en session_state
                st.session_state.predictions = predictions
                st.session_state.probabilities = probabilities
                st.session_state.agreement_metrics = agreement_metrics
                st.session_state.probability_distances = probability_distances
                st.session_state.analyzed_image = image
                st.session_state.analysis_completed = True
        
        # Verificar si hay resultados en session_state
        if hasattr(st.session_state, 'analysis_completed') and st.session_state.analysis_completed:
            predictions = st.session_state.predictions
            probabilities = st.session_state.probabilities
            agreement_metrics = st.session_state.agreement_metrics
            probability_distances = st.session_state.probability_distances
            analyzed_image = st.session_state.analyzed_image
            
            # Mostrar resultados
            st.markdown('<h2 class="sub-header">📊 Resultados del Análisis</h2>', unsafe_allow_html=True)
            
            # Resumen ejecutivo
            valid_preds = {k: v for k, v in predictions.items() if v is not None}
            valid_probs = {k: v for k, v in probabilities.items() if v is not None}
            
            if valid_probs:
                # Calcular consenso
                avg_probs = np.mean([probs for probs in valid_probs.values()], axis=0)
                consensus_class = np.argmax(avg_probs)
                consensus_confidence = avg_probs[consensus_class]
                
                # Mostrar consenso
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                    <h3>🎯 Clasificación Consenso</h3>
                    <h2>{CLASS_NAMES[consensus_class]}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                    <h3>📈 Confianza Promedio</h3>
                    <h2>{consensus_confidence:.1%}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    agreement_status = "✅ Sí" if agreement_metrics.get('all_models_agree', False) else "⚠️ No"
                    st.markdown(f"""
                    <div class="metric-card">
                    <h3>🤝 Concordancia Total</h3>
                    <h2>{agreement_status}</h2>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Resultados detallados por modelo
            st.markdown("### 🔍 Resultados Detallados por Modelo")
            
            results_cols = st.columns(2)
            
            for idx, (model_name, pred_class) in enumerate(valid_preds.items()):
                col_idx = idx % 2
                
                with results_cols[col_idx]:
                    probs = valid_probs[model_name]
                    confidence = probs[pred_class]
                    
                    st.markdown(f"""
                    <div class="model-card">
                    <h4>{model_name}</h4>
                    <p><strong>Predicción:</strong> {CLASS_NAMES[pred_class]}</p>
                    <p><strong>Confianza:</strong> {confidence:.1%}</p>
                    <p><strong>Descripción:</strong> {models[model_name]['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Mostrar todas las probabilidades
                    prob_df = pd.DataFrame({
                        'Clase': CLASS_NAMES,
                        'Probabilidad': [f"{p:.1%}" for p in probs]
                    })
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)
            
            # Visualizaciones
            st.markdown("### 📈 Visualizaciones")
            
            # Crear gráficos de predicción
            fig1, fig2 = create_prediction_visualization(probabilities, predictions)
            
            if fig1 and fig2:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    st.plotly_chart(fig2, use_container_width=True)
            
            # Análisis de concordancia y distancias
            st.markdown("### 🔗 Análisis de Concordancia y Distancias")
            
            # Crear gráficos de concordancia
            fig3, fig4 = create_agreement_visualization(agreement_metrics, probability_distances)
            
            if fig3 and fig4:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(fig3, use_container_width=True)
                
                with col2:
                    st.plotly_chart(fig4, use_container_width=True)
            
            # Métricas estadísticas detalladas
            st.markdown("### 📊 Métricas Estadísticas Detalladas")
            
            # Tabla de concordancia
            if agreement_metrics:
                st.markdown("#### 🤝 Análisis de Concordancia entre Modelos")
                
                agreement_data = []
                for comparison, metrics in agreement_metrics.items():
                    if comparison != 'all_models_agree':
                        models_pair = comparison.split('_vs_')
                        agreement_data.append({
                            'Modelo 1': models_pair[0],
                            'Modelo 2': models_pair[1],
                            'Predicción 1': CLASS_NAMES[metrics['pred1']],
                            'Predicción 2': CLASS_NAMES[metrics['pred2']],
                            'Concordancia': '✅ Sí' if metrics['agreement'] else '❌ No'
                        })
                
                if agreement_data:
                    df_agreement = pd.DataFrame(agreement_data)
                    st.dataframe(df_agreement, use_container_width=True, hide_index=True)
            
            # Tabla de distancias
            if probability_distances:
                st.markdown("#### 📏 Distancias entre Distribuciones de Probabilidad")
                
                distance_data = []
                for comparison, distances in probability_distances.items():
                    models_pair = comparison.split('_vs_')
                    distance_data.append({
                        'Modelo 1': models_pair[0],
                        'Modelo 2': models_pair[1],
                        'Distancia Euclidiana': f"{distances['euclidean']:.4f}",
                        'Distancia Coseno': f"{distances['cosine']:.4f}",
                        'Divergencia KL': f"{distances['kl_divergence']:.4f}"
                    })
                
                if distance_data:
                    df_distances = pd.DataFrame(distance_data)
                    st.dataframe(df_distances, use_container_width=True, hide_index=True)
                    
                    st.markdown("""
                    <div class="info-box">
                    <strong>💡 Interpretación de Distancias:</strong><br>
                    • <strong>Distancia Euclidiana:</strong> Distancia geométrica entre vectores de probabilidad (menor = mayor similitud)<br>
                    • <strong>Distancia Coseno:</strong> Similitud angular entre vectores (menor = mayor similitud)<br>
                    • <strong>Divergencia KL:</strong> Medida de diferencia entre distribuciones (menor = mayor similitud)
                    </div>
                    """, unsafe_allow_html=True)
            
            # Interpretación clínica
            st.markdown("### 🏥 Interpretación Clínica")
            
            if valid_probs:
                st.markdown(f"""
                <div class="success-box">
                <h4>📋 Resumen Diagnóstico</h4>
                <p><strong>Clasificación principal:</strong> {CLASS_NAMES[consensus_class]}</p>
                <p><strong>Descripción:</strong> {CLASS_DESCRIPTIONS[CLASS_NAMES[consensus_class]]}</p>
                <p><strong>Confianza del consenso:</strong> {consensus_confidence:.1%}</p>
                <p><strong>Concordancia entre modelos:</strong> {"Alta" if agreement_metrics.get('all_models_agree', False) else "Parcial"}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Recomendaciones
                if consensus_class == 0:  # Non-Tumor
                    recommendation = "👍 El análisis sugiere ausencia de tejido tumoral. Se recomienda seguimiento rutinario según protocolo clínico."
                elif consensus_class == 1:  # Non-Viable-Tumor
                    recommendation = "⚡ Se detecta tejido tumoral no viable (necrótico). Evaluar respuesta al tratamiento previo y considerar ajustes terapéuticos."
                elif consensus_class == 2:  # Viable
                    recommendation = "⚠️ Se detecta tejido tumoral viable activo. Considerar opciones de tratamiento inmediato según guidelines oncológicos."
                else:  # Mixed
                    recommendation = "🔄 Se detecta tejido mixto con características heterogéneas. Se recomienda análisis histopatológico adicional y evaluación multidisciplinaria."
                
                st.markdown(f"""
                <div class="warning-box">
                <h4>💊 Recomendación Clínica</h4>
                <p>{recommendation}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Exportar PDF
            st.markdown("### 📄 Exportar Reporte")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if st.button("📥 Generar Reporte PDF", type="secondary", use_container_width=True):
                    with st.spinner('📝 Generando reporte PDF...'):
                        try:
                            # Verificar que tenemos datos válidos
                            valid_preds = {k: v for k, v in predictions.items() if v is not None}
                            
                            if not valid_preds:
                                st.error("❌ No hay predicciones válidas para generar el reporte")
                            else:
                                pdf_buffer = generate_pdf_report(
                                    analyzed_image, predictions, probabilities, 
                                    agreement_metrics, probability_distances
                                )
                                
                                if pdf_buffer:
                                    # Verificar que el buffer tiene contenido
                                    pdf_data = pdf_buffer.getvalue()
                                    if len(pdf_data) > 0:
                                        st.download_button(
                                            label="⬇️ Descargar Reporte PDF",
                                            data=pdf_data,
                                            file_name=f"reporte_osteosarcoma_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                            mime="application/pdf",
                                            use_container_width=True
                                        )
                                        st.success("✅ Reporte PDF generado exitosamente")
                                        st.info(f"📊 Tamaño del archivo: {len(pdf_data) / 1024:.1f} KB")
                                    else:
                                        st.error("❌ Error: El archivo PDF está vacío")
                                else:
                                    st.error("❌ Error al generar el buffer del PDF")
                                    
                        except Exception as pdf_error:
                            st.error(f"❌ Error al generar el PDF: {str(pdf_error)}")
                            st.warning("💡 Intente recargar la página y volver a subir la imagen")
                            
                            # Mostrar detalles del error en modo debug
                            if st.checkbox("🔍 Mostrar detalles del error"):
                                st.code(str(pdf_error))
            
            with col2:
                if st.button("🔄 Nuevo Análisis", use_container_width=True):
                    # Limpiar session_state
                    for key in ['predictions', 'probabilities', 'agreement_metrics', 
                               'probability_distances', 'analyzed_image', 'analysis_completed']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
            
            # Disclaimer médico
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); padding: 1rem; border-radius: 10px; margin: 2rem 0;">
            <h4>⚠️ Importante - Disclaimer Médico</h4>
            <p>Este sistema es una <strong>herramienta de apoyo diagnóstico</strong> basada en inteligencia artificial. 
            Los resultados presentados <strong>NO constituyen un diagnóstico médico definitivo</strong> y deben ser 
            interpretados exclusivamente por profesionales médicos cualificados.</p>
            
            <p>Se recomienda encarecidamente:</p>
            <ul>
            <li>Correlacionar estos resultados con estudios clínicos, radiológicos e histopatológicos adicionales</li>
            <li>Considerar el contexto clínico completo del paciente</li>
            <li>Seguir las guidelines y protocolos médicos establecidos</li>
            <li>Buscar opinión de especialistas en oncología y patología cuando sea apropiado</li>
            </ul>
            
            <p><strong>Este análisis no sustituye el juicio clínico profesional.</strong></p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # No hay imagen cargada
        st.markdown("""
        <div class="info-box">
        <h4>👆 Instrucciones</h4>
        <p>Para comenzar el análisis, por favor:</p>
        <ol>
        <li>📤 <strong>Carga una imagen</strong> histopatológica usando el selector de archivos arriba</li>
        <li>🚀 <strong>Presiona "Iniciar Análisis"</strong> para procesar la imagen con los 4 modelos de IA</li>
        <li>📊 <strong>Revisa los resultados</strong> detallados y las métricas estadísticas</li>
        <li>📄 <strong>Descarga el reporte PDF</strong> con todos los hallazgos</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Mostrar información adicional cuando no hay imagen
        st.markdown("### 📋 Información del Sistema")
        
        st.markdown("""
        <div class="warning-box">
        <h4>🔬 Capacidades del Sistema</h4>
        <p>Este sistema puede analizar imágenes histopatológicas de osteosarcoma utilizando 4 modelos de deep learning diferentes:</p>
        <ul>
        <li><strong>VGG16:</strong> Arquitectura clásica profunda para análisis detallado</li>
        <li><strong>ResNet50:</strong> Red residual con conexiones skip para mejor precisión</li>
        <li><strong>MobileNetV2:</strong> Modelo eficiente optimizado para velocidad</li>
        <li><strong>EfficientNetB0:</strong> Balance óptimo entre precisión y eficiencia</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
        <h4>🎯 Clases de Clasificación</h4>
        <ul>
        <li><strong>Non-Tumor:</strong> Tejido sin presencia de tumor</li>
        <li><strong>Non-Viable-Tumor:</strong> Tejido tumoral no viable (necrótico)</li>
        <li><strong>Viable:</strong> Tejido tumoral viable (activo)</li>
        <li><strong>Mixed:</strong> Tejido mixto con características combinadas</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
