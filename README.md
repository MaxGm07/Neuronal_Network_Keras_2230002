# 💡 Red Neuronal con NumPy 🧩

![Estado](https://img.shields.io/badge/Estado-Completado-brightgreen)

:octocat:

**Estudiante**: Carlos Maximiliano García Medina

**Tarea 2**: Red Neuronal con Keras

**Materia**: Sistemas de Visión Artificial  


---

## 📝 Descripción

Este repositorio contiene la **Tarea 2** de una **red neuronal desde cero** utilizando **Keras** (librería de TensorFlow) para clasificar imágenes del conjunto de datos MNIST. El proyecto incluye:

* 🖼️ **Carga y visualización de imágenes**: Se cargan y muestran imágenes de dígitos escritos a mano del conjunto de datos MNIST.
* 🧩 **Preprocesamiento de datos**: Las imágenes se aplanan y normalizan, y las etiquetas se convierten a un formato one-hot encoding para preparar los datos para el entrenamiento.
* 🛠️ **Construcción de la red neuronal**: Se implementa una red neuronal con una capa oculta de 512 neuronas y una capa de salida de 10 neuronas (una para cada dígito del 0 al 9).
* 🚀 **Entrenamiento y evaluación**: La red neuronal se entrena durante 10 épocas y se evalúa su rendimiento en un conjunto de prueba, mostrando la precisión y la pérdida.

El código está comentado paso a paso para una mayor comprensión.

---

## 📋 Requisitos

Para ejecutar este proyecto, necesitas tener instaladas lo siguiente:
- **Python**: 
- [Python 3.9 hasta Python 3.12](https://www.python.org/downloads/)

Se puede comprobar la versión de python empleando el comando en terminal:

**IMPORTANTE:** Se requiere instalar ese rango de versiones debido a que tensorflow solo se puede emplear en esas versiones de python (https://www.tensorflow.org/install/pip?hl=es).

**En PowerShell:**

  python --version

**En Unix**

  python3 --version


Librerías (dependencias):
* NumPy: Para cálculos numéricos y manejo de arreglos.
* Matplotlib: Para la generación de gráficas.
* TensorFlow/Keras: Para construir y entrenar la red neuronal.

Puedes instalar en conjunto estas dependencias utilizando `pip`:

```bash
pip install numpy matplotlib tensorflow
```
**Nota:** Si en Unix (Linux) no funciona, emplea ```pip3```

## 🗂️ Estructura del Proyecto
El proyecto está organizado de la siguiente manera:

``` bash
Neuronal_Network_keras/
│
├── src/
│   └── Neuronal_Network_Keras.py  # Script principal de la red neuronal
│
├── .gitignore      # Archivo para ignorar archivos no deseados
├── main.py         # Script principal para ejecutar el proyecto
├── README.md       # Este archivo
└── requirements.txt # Lista de dependencias del proyecto
```
## 🚀 ¿Cómo usar este repositorio?
Sigue estos pasos para ejecutar el proyecto en tu lab:

### Clona el repositorio 🖥️:
Abre una terminal y ejecuta el siguiente comando para clonar el repositorio en tu computadora:

```bash
git clone https://github.com/MaxGm07/Neuronal_Network_Numpy_2230002
```
### Cree un nuevo entorno virtual
Se recomienda tener el entorno virtual generado en la carpeta principal para un fácil acceso, su activación y desactivación se realiza de la siguiente forma:

En PowerShell:
```
.\nombre_del_entorno\Scripts\Activate
deactivate
```
En Unix:
```
source nombre_del_entorno/bin/activate
deactivate
```
### Instala las dependencias 📦:
Asegúrate de tener instaladas las bibliotecas necesarias. Ejecuta el siguiente comando para instalarlas:

```bash
pip install -r requirements.txt
```
### Ejecuta el script principal🚀:
Para entrenar y evaluar la red neuronal, ejecuta:

```bash
python main.py
```
### Visualiza los resultados 📊:

  * El script mostrará la precisión y pérdida durante el entrenamiento en la consola.

  * También se mostrará una imagen de ejemplo del conjunto de datos.

## 🛠️ Tecnologías Utilizadas
**Python**: Lenguaje de programación principal en este caso se utilizó la versión 3.11 para el desarrollo del proyecto. (recordatorio de usar versiones 3.11-3.12)

* TensorFlow/Keras: Para construir y entrenar la red neuronal.

* NumPy: Para manejar operaciones numéricas.

* Matplotlib: Para visualizar imágenes y gráficos.

## Explicación del código
El código realiza lo siguiente:

# 📊 **Explicación del Código: Red Neuronal en Keras para MNIST**

## 📊 **Importación de Librerías**
```python
import numpy as np  
import matplotlib.pyplot as plt  
import tensorflow as tf  
from keras.models import Sequential  
from keras.layers import Dense, Input   
from keras.utils import to_categorical  
from keras.datasets import mnist  
```

**Descripción:**
- `numpy`: Manejo de arreglos numéricos.
- `matplotlib.pyplot`: Visualización de imágenes.
- `tensorflow`: Backend para la ejecución de la red neuronal.
- `keras.models.Sequential`: Construcción de la red neuronal de forma secuencial.
- `keras.layers.Dense, Input`: Define capas de la red neuronal.
- `keras.utils.to_categorical`: Convierte etiquetas en **one-hot encoding**.
- `keras.datasets.mnist`: Proporciona el dataset MNIST.

---

## 📚 **Carga del Dataset MNIST**
```python
(train_data_x, train_labels_y), (test_data_x, test_labels_y) = mnist.load_data()
```

El dataset MNIST contiene **imágenes de dígitos manuscritos** del 0 al 9:
- **60,000 imágenes** para entrenamiento.
- **10,000 imágenes** para prueba.
- Cada imagen tiene **28×28 píxeles**, con valores entre **0 y 255**.

---

## 🎨 **Visualización de una Imagen**
```python
plt.imshow(train_data_x[10], cmap='gray')  
plt.title("Ejemplo de una Imagen de Entrenamiento")  
plt.show()
```

Muestra la imagen número **10** del dataset en **escala de grises**.

---

## 🎫 **Construcción de la Red Neuronal**
```python
model = Sequential([
    Input(shape=(28 * 28,)),  
    Dense(512, activation='relu'),  
    Dense(10, activation='softmax')  
])
```

**Estructura de la red:**
- **Capa de entrada**: Recibe **784 valores** (28x28 aplanados).
- **Capa oculta**: **512 neuronas** con activación **ReLU**.
- **Capa de salida**: **10 neuronas** con activación **softmax** (una para cada dígito).



## ⚙️ **Compilación del Modelo**
```python
model.compile(
    optimizer='rmsprop',  
    loss='categorical_crossentropy',  
    metrics=['accuracy']  
)
```

**Parámetros:**
- **`optimizer='rmsprop'`**: Algoritmo de optimización.
- **`loss='categorical_crossentropy'`**: Función de pérdida para clasificación multiclase.
- **`metrics=['accuracy']`**: Métrica de rendimiento.



## 🌍 **Resumen de la Red**
```python
model.summary()
```
Muestra información de las capas, parámetros y conexiones.



## 🔄 **Preprocesamiento de Datos**
```python
x_train = train_data_x.reshape(60000, 28 * 28).astype('float32') / 255  
y_train = to_categorical(train_labels_y)  

x_test = test_data_x.reshape(10000, 28 * 28).astype('float32') / 255  
y_test = to_categorical(test_labels_y)  
```

Pasos:
1. **Aplanado**: Convierte las imágenes 28×28 en vectores de **784 valores**.
2. **Normalización**: Escala los píxeles a valores entre **0 y 1**.
3. **Conversión de etiquetas** a **one-hot encoding**. 
*(La codificación one-hot (OHE) es un método que representa variables categóricas como vectores binarios. Se utiliza para mejorar la predicción en aplicaciones de aprendizaje automático.)*



## 🏋️ **Entrenamiento del Modelo**
```python
model.fit(x_train, y_train, epochs=10, batch_size=128)
```

**Parámetros:**
- **`epochs=10`**: Número de iteraciones completas sobre el dataset.
- **`batch_size=128`**: Tamaño de los lotes de datos procesados en cada actualización de pesos.



## 🔮 **Evaluación del Modelo**
```python
loss, accuracy = model.evaluate(x_test, y_test)  
print(f"Pérdida: {loss}, Precisión: {accuracy}")
```

**Salida esperada:**
```
Evaluando la red neuronal...
Pérdida: 0.08, Precisión: 0.98
```

El modelo logra una **precisión del 98%** en la clasificación de dígitos manuscritos.

---