# 🤖 ASL Fingerspelling Recognition Project

Sistema de reconocimiento de deletreo manual en Lenguaje de Señas Americano (ASL) usando arquitectura Transformer.

## 🎯 Resumen del Problema

### ❓ ¿Qué significa este error?

El error indica que la aplicación está intentando cargar un modelo de machine learning que aún no existe. Específicamente:

- **Archivo faltante**: `models/config.pkl` (configuración del modelo)
- **Causa**: El modelo Transformer no ha sido entrenado todavía
- **Impacto**: Las secciones de "Clasificación" y "Comparación de Modelos" no funcionan

## ✅ Solución Implementada

### 🛠️ **Modo Demostración Inteligente**

Hemos implementado un **modo demostración completo** que permite usar la aplicación sin necesidad del modelo real:

#### **1. Detección Automática** 
- La app detecta automáticamente si el modelo existe
- Si no existe, activa el modo demostración
- Muestra información clara sobre por qué no hay modelo

#### **2. Simulación Realista**
- **Predicciones simuladas** con errores controlados
- **Métricas realistas** (CER, confianza, accuracy)
- **Interfaz idéntica** al modelo real

#### **3. Información Educativa**
- Explicación sobre cómo obtener el modelo real
- Métricas esperadas del modelo entrenado
- Gráficas simuladas de entrenamiento

## 🎨 Mejoras de Diseño

### **Modo Oscuro Profesional**
- ✨ **Glassmorphism effects** con transparencias
- 🌈 **Gradientes dinámicos** en títulos y botones
- 🎛️ **Toggle de tema** en la sidebar (🌙/☀️)
- 💫 **Animaciones suaves** y efectos hover
- 🎨 **Paleta de colores moderna** (GitHub Dark inspired)

### **Características Visuales**
- **Tarjetas flotantes** con sombras dinámicas
- **Efectos neón** en elementos de predicción  
- **Botones con shimmer** effect al hacer hover
- **Scrollbar personalizado** que coincide con el tema
- **Footer moderno** con información del proyecto

## 🚀 Cómo Usar la Aplicación

### **📊 Exploración de Datos**
- Dashboard interactivo con datos de demostración
- Filtros dinámicos y visualizaciones enlazadas
- Análisis de distribuciones y estadísticas

### **🤖 Clasificación (Modo Demo)**
1. Selecciona una frase de ejemplo
2. Ajusta la tasa de error simulada
3. Ejecuta la simulación
4. Ve resultados con métricas realistas

### **📈 Comparación de Modelos (Modo Demo)**
- Gráficas simuladas de entrenamiento
- Métricas esperadas del modelo real
- Información sobre arquitectura Transformer

## 🔄 Para Obtener el Modelo Real

Si quieres entrenar el modelo real necesitas:

### **📂 1. Descargar Dataset**
```bash
# Desde Kaggle: ASL Fingerspelling Recognition
data/
├── train.csv                           # Metadatos
├── train_landmarks/                    # Coordenadas (.parquet)
└── character_to_prediction_index.json  # Mapeo de caracteres
```

### **🚂 2. Entrenar Modelo**
```bash
python train_asl.py \
  --csv_path data/train.csv \
  --landmarks_path data/train_landmarks \
  --char_mapping data/character_to_prediction_index.json \
  --epochs 13 \
  --batch_size 64
```

### **⏱️ 3. Tiempo Estimado**
- **GPU recomendada**: 2-4 horas
- **CPU únicamente**: 8-12 horas
- **RAM mínima**: 8GB
- **Almacenamiento**: 5GB libres

### **📁 4. Estructura Final**
```
models/
├── asl_transformer/          # Modelo completo
├── asl_transformer_weights.h5
├── config.pkl               # ✅ Archivo que faltaba
├── history.pkl              # Historia de entrenamiento  
└── training_results.png     # Gráficas
```

## 🎯 Arquitectura del Modelo

### **🧠 Transformer Encoder-Decoder**
- **Encoder**: 2 capas + embedding convolucional
- **Decoder**: 1 capa + atención causal
- **Embedding dimension**: 200
- **Attention heads**: 4
- **Feed-forward**: 400 dimensiones

### **📊 Métricas Esperadas**
- **CER final**: ~15% (85% precisión)
- **Confianza promedio**: 85-95%
- **Vocabulario**: 62 caracteres (a-z, 0-9, espacio, tokens especiales)
- **Input**: 128 frames de landmarks (63 coordenadas x,y,z por frame)
- **Output**: Secuencia de hasta 64 caracteres

## 🛠️ Tecnologías Utilizadas

- **🐍 Python 3.8+**
- **🚀 Streamlit** (interfaz web)
- **🧠 TensorFlow 2.x** (modelo ML)  
- **📊 Plotly** (visualizaciones)
- **🐼 Pandas/NumPy** (datos)
- **⚡ PyArrow** (parquet files)

## 📱 Uso de la Aplicación

### **Ejecutar Localmente**
```bash
# 1. Activar entorno virtual
source .venv/bin/activate

# 2. Instalar dependencias (si no están)
pip install streamlit tensorflow pandas plotly pyarrow python-Levenshtein

# 3. Ejecutar aplicación
streamlit run app.py
```

### **URLs de Acceso**
- **Local**: http://localhost:8501
- **Red local**: http://[tu-ip]:8501

## 🎮 Controles de la Aplicación

### **🎛️ Sidebar**
- **🌙/☀️**: Toggle modo oscuro/claro
- **🎨**: Selector de tema para gráficas
- **📊**: Paleta de colores actual

### **📍 Navegación**
- **🏠 Inicio**: Overview del proyecto
- **📊 Exploración**: Dashboard de datos
- **🤖 Clasificación**: Demo de predicciones
- **📈 Comparación**: Métricas y gráficas
- **ℹ️ Acerca**: Información técnica

## 🤝 Contribuir

Este es un proyecto educativo. Para contribuir:

1. Fork el repositorio
2. Crea una rama feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit cambios (`git commit -m 'Agregar nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Abre un Pull Request

## 📄 Licencia

Proyecto educativo para aprendizaje de machine learning y desarrollo web.

## 🙏 Reconocimientos

- **Kaggle**: Por el dataset ASL Fingerspelling
- **Google Research**: Por MediaPipe y la extracción de landmarks
- **TensorFlow/Keras**: Por las herramientas de deep learning
- **Streamlit**: Por la plataforma de desarrollo de apps

---

💡 **Tip**: La aplicación funciona perfectamente en modo demostración. ¡Explora todas las funcionalidades sin necesidad de datos reales!