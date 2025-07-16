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
from i18n import get_text, get_class_names, get_class_descriptions, get_model_descriptions, get_available_languages, get_language_names
warnings.filterwarnings('ignore')

# Initialize session state for language
if 'language' not in st.session_state:
    st.session_state.language = 'es'  # Default to Spanish

# Configuración de la página
st.set_page_config(
    page_title=get_text('page_title', st.session_state.language),
    page_icon=get_text('page_icon', st.session_state.language),
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
    
    for model_name in ['VGG16', 'ResNet50', 'MobileNetV2', 'EfficientNetB0']:
        try:
            model_path = f'models/{model_name}_osteosarcoma.h5'
            models[model_name] = {
                'model': load_model(model_path),
                'preprocess': {
                    'VGG16': vgg_preprocess,
                    'ResNet50': resnet_preprocess,
                    'MobileNetV2': mobilenet_preprocess,
                    'EfficientNetB0': efficientnet_preprocess
                }[model_name]
            }
        except Exception as e:
            st.error(f"{get_text('error_loading_model', st.session_state.language)} {model_name}: {str(e)}")
    
    return models

# Configuración de clases - ahora usando i18n
def get_current_class_names():
    """Get class names for current language"""
    return get_class_names(st.session_state.language)

def get_current_class_descriptions():
    """Get class descriptions for current language"""
    return get_class_descriptions(st.session_state.language)

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
    class_names = get_current_class_names()
    
    for model_name, probs in valid_probs.items():
        for i, class_name in enumerate(class_names):
            prob_data.append({
                get_text('models', st.session_state.language): model_name,
                get_text('class', st.session_state.language): class_name,
                get_text('probability', st.session_state.language): probs[i],
                get_text('prediction', st.session_state.language): predictions[model_name] == i
            })
    
    df_probs = pd.DataFrame(prob_data)
    
    # Gráfico de barras comparativo
    fig1 = px.bar(
        df_probs, 
        x=get_text('class', st.session_state.language), 
        y=get_text('probability', st.session_state.language), 
        color=get_text('models', st.session_state.language),
        title=get_text('probability_distribution', st.session_state.language),
        labels={
            get_text('probability', st.session_state.language): get_text('probability', st.session_state.language) + ' (%)', 
            get_text('class', st.session_state.language): get_text('tissue_type', st.session_state.language)
        },
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
    prob_matrix = df_probs.pivot(
        index=get_text('models', st.session_state.language), 
        columns=get_text('class', st.session_state.language), 
        values=get_text('probability', st.session_state.language)
    )
    
    fig2 = px.imshow(
        prob_matrix.values,
        x=prob_matrix.columns,
        y=prob_matrix.index,
        title=get_text('probability_heatmap', st.session_state.language),
        aspect='auto',
        color_continuous_scale='viridis',
        labels={get_text('color', st.session_state.language): get_text('probability', st.session_state.language)},
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
    
    class_names = get_current_class_names()
    
    for comparison, metrics in agreement_metrics.items():
        if comparison != 'all_models_agree':
            models = comparison.split('_vs_')
            agreement_data.append({
                get_text('comparison', st.session_state.language): f"{models[0]} vs {models[1]}",
                get_text('concordance', st.session_state.language): get_text('yes', st.session_state.language) if metrics['agreement'] else get_text('no', st.session_state.language),
                get_text('prediction_1', st.session_state.language): class_names[metrics['pred1']],
                get_text('prediction_2', st.session_state.language): class_names[metrics['pred2']]
            })
    
    # Preparar datos de distancias
    for comparison, distances in probability_distances.items():
        models = comparison.split('_vs_')
        distance_data.append({
            get_text('comparison', st.session_state.language): f"{models[0]} vs {models[1]}",
            get_text('euclidean_distance', st.session_state.language): distances['euclidean'],
            get_text('cosine_distance', st.session_state.language): distances['cosine'],
            get_text('kl_divergence', st.session_state.language): distances['kl_divergence']
        })
    
    # Gráfico de concordancia
    df_agreement = pd.DataFrame(agreement_data)
    
    fig1 = px.bar(
        df_agreement,
        x=get_text('comparison', st.session_state.language),
        color=get_text('concordance', st.session_state.language),
        title=get_text('model_agreement', st.session_state.language),
        labels={'count': get_text('number_of_comparisons', st.session_state.language)},
        template='plotly_white',
        height=400
    )
    
    fig1.update_layout(title_x=0.5, xaxis_tickangle=-45)
    
    # Gráfico de distancias
    df_distances = pd.DataFrame(distance_data)
    
    fig2 = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            get_text('euclidean_distance', st.session_state.language), 
            get_text('cosine_distance', st.session_state.language), 
            get_text('kl_divergence', st.session_state.language)
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Añadir barras para cada métrica de distancia
    for i, metric in enumerate([
        get_text('euclidean_distance', st.session_state.language),
        get_text('cosine_distance', st.session_state.language),
        get_text('kl_divergence', st.session_state.language)
    ]):
        fig2.add_trace(
            go.Bar(
                x=df_distances[get_text('comparison', st.session_state.language)],
                y=df_distances[metric],
                name=metric,
                showlegend=False
            ),
            row=1, col=i+1
        )
    
    fig2.update_layout(
        title_text=get_text('probability_distances', st.session_state.language),
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
        
        # Get current language settings
        lang = st.session_state.language
        class_names = get_current_class_names()
        class_descriptions = get_current_class_descriptions()
        
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
        story.append(Paragraph(get_text('pdf_report_title', lang), title_style))
        story.append(Spacer(1, 30))
        
        # Información general
        story.append(Paragraph(get_text('analysis_info', lang), heading_style))
        story.append(Paragraph(f"<b>{get_text('analysis_date', lang)}:</b> {pd.Timestamp.now().strftime(get_text('date_format', lang))}", normal_style))
        story.append(Paragraph(f"<b>{get_text('models_used', lang)}:</b> VGG16, ResNet50, MobileNetV2, EfficientNetB0", normal_style))
        story.append(Paragraph(f"<b>{get_text('classes_analyzed', lang)}:</b> {', '.join(class_names)}", normal_style))
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
            story.append(Paragraph(get_text('analyzed_image', lang), heading_style))
            
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
                story.append(Paragraph(f"<i>{get_text('error_inserting_image', lang)}: {str(img_error)}</i>", normal_style))
                story.append(Spacer(1, 20))
        
        # Resultados de predicción
        story.append(Paragraph(get_text('prediction_results', lang), heading_style))
        
        valid_preds = {k: v for k, v in predictions.items() if v is not None}
        valid_probs = {k: v for k, v in probabilities.items() if v is not None}
        
        if valid_preds:
            for model_name in valid_preds.keys():
                pred_class = valid_preds[model_name]
                pred_probs = valid_probs[model_name]
                
                story.append(Paragraph(f"<b>{model_name}:</b>", normal_style))
                story.append(Paragraph(f"• {get_text('prediction', lang)}: <b>{class_names[pred_class]}</b>", normal_style))
                story.append(Paragraph(f"• {get_text('confidence_label', lang)}: <b>{pred_probs[pred_class]:.1%}</b>", normal_style))
                
                # Añadir todas las probabilidades
                prob_text = f"• {get_text('probability_distribution_label', lang)}:<br/>"
                for i, class_name in enumerate(class_names):
                    prob_text += f"&nbsp;&nbsp;&nbsp;&nbsp;- {class_name}: {pred_probs[i]:.1%}<br/>"
                
                story.append(Paragraph(prob_text, normal_style))
                story.append(Spacer(1, 10))
        
        # Análisis de concordancia
        story.append(Paragraph(get_text('agreement_analysis_pdf', lang), heading_style))
        
        if agreement_metrics:
            if agreement_metrics.get('all_models_agree', False):
                story.append(Paragraph(f"✅ <b>{get_text('all_models_agree', lang)}</b>", normal_style))
            else:
                story.append(Paragraph(f"⚠️ <b>{get_text('models_disagree', lang)}</b>", normal_style))
            
            # Detalles de concordancia por pares
            story.append(Paragraph(f"<b>{get_text('pair_details', lang)}:</b>", normal_style))
            for comparison, metrics in agreement_metrics.items():
                if comparison != 'all_models_agree':
                    models = comparison.split('_vs_')
                    agreement_text = get_text('agree', lang) if metrics['agreement'] else get_text('disagree', lang)
                    story.append(Paragraph(
                        f"• {models[0]} {get_text('vs', lang)} {models[1]}: <b>{agreement_text}</b> "
                        f"({class_names[metrics['pred1']]} {get_text('vs', lang)} {class_names[metrics['pred2']]})",
                        normal_style
                    ))
        
        story.append(Spacer(1, 20))
        
        # Distancias entre modelos
        if probability_distances:
            story.append(Paragraph(get_text('probability_distances_pdf', lang), heading_style))
            
            for comparison, distances in probability_distances.items():
                models = comparison.split('_vs_')
                story.append(Paragraph(f"<b>{models[0]} {get_text('vs', lang)} {models[1]}:</b>", normal_style))
                story.append(Paragraph(f"• {get_text('euclidean_distance', lang)}: {distances['euclidean']:.4f}", normal_style))
                story.append(Paragraph(f"• {get_text('cosine_distance', lang)}: {distances['cosine']:.4f}", normal_style))
                story.append(Paragraph(f"• {get_text('kl_divergence', lang)}: {distances['kl_divergence']:.4f}", normal_style))
                story.append(Spacer(1, 10))
        
        # Interpretación médica
        story.append(Paragraph(get_text('clinical_interpretation_pdf', lang), heading_style))
        
        # Determinar la clase más probable
        if valid_probs:
            avg_probs = np.mean([probs for probs in valid_probs.values()], axis=0)
            most_likely_class = np.argmax(avg_probs)
            confidence = avg_probs[most_likely_class]
            
            story.append(Paragraph(f"<b>{get_text('consensus_classification_pdf', lang)}:</b> {class_names[most_likely_class]}", normal_style))
            story.append(Paragraph(f"<b>{get_text('average_confidence_pdf', lang)}:</b> {confidence:.1%}", normal_style))
            story.append(Paragraph(f"<b>{get_text('description_pdf', lang)}:</b> {class_descriptions[class_names[most_likely_class]]}", normal_style))
            
            # Recomendaciones basadas en el resultado
            recommendations = {
                0: get_text('recommendation_non_tumor', lang),
                1: get_text('recommendation_non_viable', lang),
                2: get_text('recommendation_viable', lang),
                3: get_text('recommendation_mixed', lang)
            }
            
            recommendation = recommendations.get(most_likely_class, "Consultar con especialista para interpretación adicional.")
            
            story.append(Spacer(1, 15))
            story.append(Paragraph(f"<b>💊 {get_text('clinical_recommendation', lang)}:</b>", normal_style))
            story.append(Paragraph(recommendation, normal_style))
        
        # Disclaimer
        story.append(Spacer(1, 30))
        story.append(Paragraph(get_text('important_note', lang), heading_style))
        
        disclaimer_text = f"""
        <b>{get_text('pdf_disclaimer_title', lang)}</b><br/><br/>
        
        {get_text('pdf_disclaimer_main', lang)}<br/><br/>
        
        <b>{get_text('pdf_disclaimer_recommendations', lang)}</b><br/>
        • {get_text('pdf_disclaimer_item_1', lang)}<br/>
        • {get_text('pdf_disclaimer_item_2', lang)}<br/>
        • {get_text('pdf_disclaimer_item_3', lang)}<br/>
        • {get_text('pdf_disclaimer_item_4', lang)}<br/><br/>
        
        <b>{get_text('pdf_disclaimer_final', lang)}</b>
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
    # Language selector in sidebar
    with st.sidebar:
        st.markdown(f"### {get_text('sidebar_language', st.session_state.language)}")
        
        # Language selector
        language_names = get_language_names()
        current_lang_name = language_names[st.session_state.language]
        
        selected_lang_name = st.selectbox(
            "",
            options=list(language_names.values()),
            index=list(language_names.values()).index(current_lang_name),
            key="language_selector"
        )
        
        # Update language if changed
        selected_lang_code = [k for k, v in language_names.items() if v == selected_lang_name][0]
        if selected_lang_code != st.session_state.language:
            st.session_state.language = selected_lang_code
            st.rerun()
    
    # Título principal
    st.markdown(f'<h1 class="main-header">{get_text("main_header", st.session_state.language)}</h1>', unsafe_allow_html=True)
    
    # Información en el sidebar
    with st.sidebar:
        st.markdown(f"### {get_text('sidebar_system_info', st.session_state.language)}")
        
        st.markdown(f"""
        <div class="info-box">
        <h4>{get_text('sidebar_objective', st.session_state.language)}</h4>
        <p>{get_text('sidebar_objective_desc', st.session_state.language)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"### {get_text('sidebar_available_models', st.session_state.language)}")
        
        model_descriptions = get_model_descriptions(st.session_state.language)
        
        for model, desc in model_descriptions.items():
            st.markdown(f"""
            <div class="model-card">
            <strong>{model}</strong><br>
            <small>{desc}</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"### {get_text('sidebar_diagnosis_classes', st.session_state.language)}")
        
        class_descriptions = get_current_class_descriptions()
        for class_name, description in class_descriptions.items():
            st.markdown(f"**{class_name}:** {description}")
    
    # Cargar modelos
    with st.spinner(get_text('loading_models', st.session_state.language)):
        models = load_models()
    
    if not models:
        st.error(get_text('error_loading_models', st.session_state.language))
        return
    
    st.success(f"✅ {len(models)} {get_text('models_loaded', st.session_state.language)}")
    
    # Sección de carga de imagen
    st.markdown(f'<h2 class="sub-header">{get_text("sub_header_upload", st.session_state.language)}</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        get_text('file_upload_label', st.session_state.language),
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        help=get_text('file_upload_help', st.session_state.language)
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
            st.image(image, caption=get_text('image_original', st.session_state.language), use_container_width=True)
            
            # Información de la imagen
            st.markdown(f"""
            <div class="info-box">
            <strong>{get_text('image_info_title', st.session_state.language)}</strong><br>
            📏 {get_text('image_dimensions', st.session_state.language)}: {image.size[0]} × {image.size[1]} {get_text('px', st.session_state.language)}<br>
            🎨 {get_text('image_mode', st.session_state.language)}: {image.mode}<br>
            📁 {get_text('image_format', st.session_state.language)}: {image.format}<br>
            💾 {get_text('image_size', st.session_state.language)}: {len(uploaded_file.getvalue()) / 1024:.1f} {get_text('kb', st.session_state.language)}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Imagen redimensionada para análisis
            resized_image = image.resize((224, 224))
            st.image(resized_image, caption=get_text('image_resized', st.session_state.language), use_container_width=True)
            
            st.markdown(f"""
            <div class="warning-box">
            <strong>⚠️ {get_text('important_note', st.session_state.language)[:4]}:</strong> {get_text('image_resize_note', st.session_state.language)}
            </div>
            """, unsafe_allow_html=True)
        
        # Botón de análisis
        if st.button(get_text('btn_start_analysis', st.session_state.language), type="primary", use_container_width=True):
            with st.spinner(get_text('loading_analysis', st.session_state.language)):
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
            st.markdown(f'<h2 class="sub-header">{get_text("sub_header_results", st.session_state.language)}</h2>', unsafe_allow_html=True)
            
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
                
                class_names = get_current_class_names()
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                    <h3>{get_text('consensus_classification', st.session_state.language)}</h3>
                    <h2>{class_names[consensus_class]}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                    <h3>{get_text('average_confidence', st.session_state.language)}</h3>
                    <h2>{consensus_confidence:.1%}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    agreement_status = get_text('agreement_yes', st.session_state.language) if agreement_metrics.get('all_models_agree', False) else get_text('agreement_no', st.session_state.language)
                    st.markdown(f"""
                    <div class="metric-card">
                    <h3>{get_text('total_agreement', st.session_state.language)}</h3>
                    <h2>{agreement_status}</h2>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Resultados detallados por modelo
            st.markdown(f"### {get_text('detailed_results', st.session_state.language)}")
            
            results_cols = st.columns(2)
            model_descriptions = get_model_descriptions(st.session_state.language)
            
            for idx, (model_name, pred_class) in enumerate(valid_preds.items()):
                col_idx = idx % 2
                
                with results_cols[col_idx]:
                    probs = valid_probs[model_name]
                    confidence = probs[pred_class]
                    
                    st.markdown(f"""
                    <div class="model-card">
                    <h4>{model_name}</h4>
                    <p><strong>{get_text('prediction', st.session_state.language)}:</strong> {class_names[pred_class]}</p>
                    <p><strong>{get_text('confidence', st.session_state.language)}:</strong> {confidence:.1%}</p>
                    <p><strong>{get_text('description', st.session_state.language)}:</strong> {model_descriptions[model_name]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Mostrar todas las probabilidades
                    prob_df = pd.DataFrame({
                        get_text('class', st.session_state.language): class_names,
                        get_text('probability', st.session_state.language): [f"{p:.1%}" for p in probs]
                    })
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)
            
            # Visualizaciones
            st.markdown(f"### {get_text('visualizations', st.session_state.language)}")
            
            # Crear gráficos de predicción
            fig1, fig2 = create_prediction_visualization(probabilities, predictions)
            
            if fig1 and fig2:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    st.plotly_chart(fig2, use_container_width=True)
            
            # Análisis de concordancia y distancias
            st.markdown(f"### {get_text('agreement_concordance', st.session_state.language)}")
            
            # Crear gráficos de concordancia
            fig3, fig4 = create_agreement_visualization(agreement_metrics, probability_distances)
            
            if fig3 and fig4:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(fig3, use_container_width=True)
                
                with col2:
                    st.plotly_chart(fig4, use_container_width=True)
            
            # Métricas estadísticas detalladas
            st.markdown(f"### {get_text('statistical_analysis', st.session_state.language)}")
            
            # Tabla de concordancia
            if agreement_metrics:
                st.markdown(f"#### {get_text('agreement_analysis', st.session_state.language)}")
                
                agreement_data = []
                for comparison, metrics in agreement_metrics.items():
                    if comparison != 'all_models_agree':
                        models_pair = comparison.split('_vs_')
                        agreement_data.append({
                            get_text('model_1', st.session_state.language): models_pair[0],
                            get_text('model_2', st.session_state.language): models_pair[1],
                            get_text('prediction_1', st.session_state.language): class_names[metrics['pred1']],
                            get_text('prediction_2', st.session_state.language): class_names[metrics['pred2']],
                            get_text('concordance', st.session_state.language): get_text('agreement_yes', st.session_state.language)[2:] if metrics['agreement'] else get_text('agreement_no', st.session_state.language)[2:]
                        })
                
                if agreement_data:
                    df_agreement = pd.DataFrame(agreement_data)
                    st.dataframe(df_agreement, use_container_width=True, hide_index=True)
            
            # Tabla de distancias
            if probability_distances:
                st.markdown(f"#### {get_text('distance_analysis', st.session_state.language)}")
                
                distance_data = []
                for comparison, distances in probability_distances.items():
                    models_pair = comparison.split('_vs_')
                    distance_data.append({
                        get_text('model_1', st.session_state.language): models_pair[0],
                        get_text('model_2', st.session_state.language): models_pair[1],
                        get_text('euclidean_distance', st.session_state.language): f"{distances['euclidean']:.4f}",
                        get_text('cosine_distance', st.session_state.language): f"{distances['cosine']:.4f}",
                        get_text('kl_divergence', st.session_state.language): f"{distances['kl_divergence']:.4f}"
                    })
                
                if distance_data:
                    df_distances = pd.DataFrame(distance_data)
                    st.dataframe(df_distances, use_container_width=True, hide_index=True)
                    
                    st.markdown(f"""
                    <div class="info-box">
                    <strong>{get_text('distance_interpretation', st.session_state.language)}</strong><br>
                    • <strong>{get_text('euclidean_distance', st.session_state.language)}:</strong> {get_text('euclidean_desc', st.session_state.language)}<br>
                    • <strong>{get_text('cosine_distance', st.session_state.language)}:</strong> {get_text('cosine_desc', st.session_state.language)}<br>
                    • <strong>{get_text('kl_divergence', st.session_state.language)}:</strong> {get_text('kl_desc', st.session_state.language)}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Interpretación clínica
            st.markdown(f"### {get_text('clinical_interpretation', st.session_state.language)}")
            
            class_descriptions = get_current_class_descriptions()
            
            if valid_probs:
                st.markdown(f"""
                <div class="success-box">
                <h4>{get_text('diagnostic_summary', st.session_state.language)}</h4>
                <p><strong>{get_text('main_classification', st.session_state.language)}:</strong> {class_names[consensus_class]}</p>
                <p><strong>{get_text('description', st.session_state.language)}:</strong> {class_descriptions[class_names[consensus_class]]}</p>
                <p><strong>{get_text('consensus_confidence', st.session_state.language)}:</strong> {consensus_confidence:.1%}</p>
                <p><strong>{get_text('model_concordance', st.session_state.language)}:</strong> {get_text('concordance_high', st.session_state.language) if agreement_metrics.get('all_models_agree', False) else get_text('concordance_partial', st.session_state.language)}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Recomendaciones
                if consensus_class == 0:  # Non-Tumor
                    recommendation = get_text('recommendation_non_tumor', st.session_state.language)
                elif consensus_class == 1:  # Non-Viable-Tumor
                    recommendation = get_text('recommendation_non_viable', st.session_state.language)
                elif consensus_class == 2:  # Viable
                    recommendation = get_text('recommendation_viable', st.session_state.language)
                else:  # Mixed
                    recommendation = get_text('recommendation_mixed', st.session_state.language)
                
                st.markdown(f"""
                <div class="warning-box">
                <h4>{get_text('clinical_recommendation', st.session_state.language)}</h4>
                <p>{recommendation}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Exportar PDF
            st.markdown(f"### {get_text('export_report', st.session_state.language)}")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if st.button(get_text('btn_generate_pdf', st.session_state.language), type="secondary", use_container_width=True):
                    with st.spinner(get_text('loading_pdf', st.session_state.language)):
                        try:
                            # Verificar que tenemos datos válidos
                            valid_preds = {k: v for k, v in predictions.items() if v is not None}
                            
                            if not valid_preds:
                                st.error(get_text('error_no_valid_predictions', st.session_state.language))
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
                                            label=get_text('btn_download_pdf', st.session_state.language),
                                            data=pdf_data,
                                            file_name=f"reporte_osteosarcoma_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                            mime="application/pdf",
                                            use_container_width=True
                                        )
                                        st.success(get_text('pdf_generated', st.session_state.language))
                                        st.info(f"📊 {get_text('file_size', st.session_state.language)}: {len(pdf_data) / 1024:.1f} {get_text('kb', st.session_state.language)}")
                                    else:
                                        st.error(get_text('error_empty_pdf', st.session_state.language))
                                else:
                                    st.error(get_text('error_pdf_buffer', st.session_state.language))
                                    
                        except Exception as pdf_error:
                            st.error(f"{get_text('error_generating_pdf', st.session_state.language)}: {str(pdf_error)}")
                            st.warning(get_text('error_reload_suggestion', st.session_state.language))
                            
                            # Mostrar detalles del error en modo debug
                            if st.checkbox(get_text('error_show_details', st.session_state.language)):
                                st.code(str(pdf_error))
            
            with col2:
                if st.button(get_text('btn_new_analysis', st.session_state.language), use_container_width=True):
                    # Limpiar session_state
                    for key in ['predictions', 'probabilities', 'agreement_metrics', 
                               'probability_distances', 'analyzed_image', 'analysis_completed']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
            
            # Disclaimer médico
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); padding: 1rem; border-radius: 10px; margin: 2rem 0;">
            <h4>{get_text('medical_disclaimer', st.session_state.language)}</h4>
            <p>{get_text('disclaimer_text', st.session_state.language)}</p>
            
            <p>{get_text('disclaimer_recommendations', st.session_state.language)}</p>
            <ul>
            <li>{get_text('disclaimer_item_1', st.session_state.language)}</li>
            <li>{get_text('disclaimer_item_2', st.session_state.language)}</li>
            <li>{get_text('disclaimer_item_3', st.session_state.language)}</li>
            <li>{get_text('disclaimer_item_4', st.session_state.language)}</li>
            </ul>
            
            <p><strong>{get_text('disclaimer_final', st.session_state.language)}</strong></p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # No hay imagen cargada
        st.markdown(f"""
        <div class="info-box">
        <h4>{get_text('instructions_title', st.session_state.language)}</h4>
        <p>{get_text('instructions_intro', st.session_state.language)}</p>
        <ol>
        <li>{get_text('instruction_1', st.session_state.language)}</li>
        <li>{get_text('instruction_2', st.session_state.language)}</li>
        <li>{get_text('instruction_3', st.session_state.language)}</li>
        <li>{get_text('instruction_4', st.session_state.language)}</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Mostrar información adicional cuando no hay imagen
        st.markdown(f"### {get_text('sidebar_system_info', st.session_state.language)}")
        
        st.markdown(f"""
        <div class="warning-box">
        <h4>{get_text('system_capabilities', st.session_state.language)}</h4>
        <p>{get_text('system_capabilities_desc', st.session_state.language)}</p>
        <ul>
        <li><strong>VGG16:</strong> {get_text('vgg16_full_desc', st.session_state.language)}</li>
        <li><strong>ResNet50:</strong> {get_text('resnet50_full_desc', st.session_state.language)}</li>
        <li><strong>MobileNetV2:</strong> {get_text('mobilenetv2_full_desc', st.session_state.language)}</li>
        <li><strong>EfficientNetB0:</strong> {get_text('efficientnetb0_full_desc', st.session_state.language)}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        class_descriptions = get_current_class_descriptions()
        st.markdown(f"""
        <div class="success-box">
        <h4>{get_text('classification_classes', st.session_state.language)}</h4>
        <ul>
        """ + ''.join([f"<li><strong>{class_name}:</strong> {desc}</li>" for class_name, desc in class_descriptions.items()]) + """
        </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
