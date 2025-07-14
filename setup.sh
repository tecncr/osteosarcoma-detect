#!/bin/bash

# Script de configuración para el Clasificador de Osteosarcoma
echo "🔬 Configurando Clasificador de Osteosarcoma..."

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 no está instalado. Por favor instala Python 3.8 o superior."
    exit 1
fi

echo "✅ Python encontrado: $(python3 --version)"

# Crear entorno virtual
echo "📦 Creando entorno virtual..."
python3 -m venv venv

# Activar entorno virtual
echo "🔄 Activando entorno virtual..."
source venv/bin/activate

# Actualizar pip
echo "⬆️ Actualizando pip..."
pip install --upgrade pip

# Instalar dependencias
echo "📚 Instalando dependencias..."
pip install -r requirements.txt

# Verificar modelos
echo "🔍 Verificando modelos..."
if [ ! -d "models" ]; then
    echo "⚠️ Carpeta 'models' no encontrada. Creando..."
    mkdir models
fi

models_required=("VGG16_osteosarcoma.h5" "ResNet50_osteosarcoma.h5" "MobileNetV2_osteosarcoma.h5" "EfficientNetB0_osteosarcoma.h5")
models_missing=()

for model in "${models_required[@]}"; do
    if [ ! -f "models/$model" ]; then
        models_missing+=("$model")
    fi
done

if [ ${#models_missing[@]} -eq 0 ]; then
    echo "✅ Todos los modelos están disponibles"
else
    echo "⚠️ Modelos faltantes:"
    for model in "${models_missing[@]}"; do
        echo "   - models/$model"
    done
    echo "Por favor, asegúrate de tener todos los modelos en la carpeta 'models/' antes de ejecutar la aplicación."
fi

echo ""
echo "🎉 ¡Configuración completada!"
echo ""
echo "Para ejecutar la aplicación:"
echo "1. Activa el entorno virtual: source venv/bin/activate"
echo "2. Ejecuta la aplicación: streamlit run app.py"
echo ""
echo "La aplicación se abrirá en tu navegador en http://localhost:8501"
