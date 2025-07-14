#!/usr/bin/env python3
"""
Script para generar una imagen de prueba para el clasificador de osteosarcoma
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_sample_image():
    """Crea una imagen de muestra para probar la aplicación"""
    
    # Crear imagen base
    width, height = 512, 512
    img = Image.new('RGB', (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Simular textura histopatológica
    np.random.seed(42)
    
    # Añadir patrones celulares simulados
    for _ in range(200):
        x = np.random.randint(0, width-20)
        y = np.random.randint(0, height-20)
        size = np.random.randint(5, 15)
        
        # Colores típicos de tinción H&E
        colors = [
            (180, 120, 160),  # Rosa (citoplasma)
            (120, 100, 180),  # Azul/morado (núcleos)
            (200, 180, 200),  # Rosa claro
            (100, 80, 140),   # Morado oscuro
        ]
        
        color = colors[np.random.randint(0, len(colors))]
        draw.ellipse([x, y, x+size, y+size], fill=color)
    
    # Añadir algunas estructuras más grandes
    for _ in range(50):
        x = np.random.randint(0, width-40)
        y = np.random.randint(0, height-40)
        size_x = np.random.randint(10, 30)
        size_y = np.random.randint(10, 30)
        
        color = (160, 140, 180)
        draw.ellipse([x, y, x+size_x, y+size_y], fill=color, outline=(100, 80, 120))
    
    # Añadir texto informativo
    try:
        # Intentar usar una fuente más grande
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        # Fallback a fuente por defecto
        font = ImageFont.load_default()
    
    # Añadir marca de agua
    draw.text((10, 10), "MUESTRA HISTOPATOLÓGICA", fill=(60, 60, 60), font=font)
    draw.text((10, height-40), "Imagen generada para pruebas", fill=(60, 60, 60), font=font)
    
    return img

def main():
    """Función principal"""
    
    # Crear directorio de ejemplos si no existe
    examples_dir = "examples"
    if not os.path.exists(examples_dir):
        os.makedirs(examples_dir)
    
    # Generar imagen de muestra
    sample_img = create_sample_image()
    
    # Guardar imagen
    sample_path = os.path.join(examples_dir, "muestra_histopatologica.png")
    sample_img.save(sample_path, "PNG")
    
    print(f"✅ Imagen de muestra creada: {sample_path}")
    print(f"📏 Dimensiones: {sample_img.size}")
    print("🔬 Esta imagen puede ser utilizada para probar la aplicación")

if __name__ == "__main__":
    main()
