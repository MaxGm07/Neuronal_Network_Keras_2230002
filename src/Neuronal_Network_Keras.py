import numpy as np  
import matplotlib.pyplot as plt  
import tensorflow as tf  
from keras.models import Sequential  # type: ignore
from keras.layers import Dense, Input   # type: ignore
from keras.utils import to_categorical  # type: ignore
from keras.datasets import mnist  # type: ignore

def E_Red_neuronal():
    # Cargamos el conjunto de datos MNIST, que tiene imágenes de dígitos escritos a mano
    (train_data_x, train_labels_y), (test_data_x, test_labels_y) = mnist.load_data()

    # Mostramos información sobre los datos de entrenamiento
    print("Forma de los datos de entrenamiento:", train_data_x.shape)  # Tamaño de las imágenes de entrenamiento (60000 imágenes de 28x28 píxeles)
    print("Etiqueta del primer ejemplo de entrenamiento:", train_labels_y[10])  # Etiqueta de la primera imagen (qué número es)
    print("Forma de los datos de prueba:", test_data_x.shape)  # Tamaño de las imágenes de prueba (10000 imágenes de 28x28 píxeles)

    # Mostramos una de las imágenes de entrenamiento para ver cómo se ve
    plt.imshow(train_data_x[10])  # Mostramos la imagen en escala de grises
    plt.title("Ejemplo de una Imagen de Entrenamiento")  # Le ponemos un título a la imagen
    plt.show()  # Mostramos la imagen en una ventana

    # Creamos la red neuronal
    model = Sequential([
        Input(shape=(28 * 28,)),  # Capa de entrada: las imágenes se aplanan a un vector de 784 valores (28x28)
        Dense(512, activation='relu'),  # Capa oculta: 512 neuronas con activación ReLU (una función que ayuda a aprender mejor)
        Dense(10, activation='softmax')  # Capa de salida: 10 neuronas (una por cada dígito del 0 al 9) con activación softmax (para obtener probabilidades)
    ])

    # Configuramos el modelo para que sepa cómo aprender
    model.compile(
        optimizer='rmsprop',  # Usamos el optimizador RMSprop para ajustar los pesos de la red
        loss='categorical_crossentropy',  # Función de pérdida: mide qué tan mal está aprendiendo la red
        metrics=['accuracy']  # Métrica: queremos medir la precisión (porcentaje de aciertos)
    )

    # Mostramos un resumen de cómo está construida la red neuronal
    print("Resumen del modelo:")
    model.summary()  # Nos dice cuántas capas tiene, cuántos parámetros, etc.

    # Preparamos los datos de entrenamiento para que la red pueda usarlos
    x_train = train_data_x.reshape(60000, 28 * 28)  # Aplanamos las imágenes de 28x28 a vectores de 784 valores
    x_train = x_train.astype('float32') / 255  # Normalizamos los valores de los píxeles para que estén entre 0 y 1
    y_train = to_categorical(train_labels_y)  # Convertimos las etiquetas a un formato especial (one-hot encoding)

    # Preparamos los datos de prueba de la misma manera
    x_test = test_data_x.reshape(10000, 28 * 28)  # Aplanamos las imágenes de prueba
    x_test = x_test.astype('float32') / 255  # Normalizamos los valores de los píxeles
    y_test = to_categorical(test_labels_y)  # Convertimos las etiquetas de prueba

    # Entrenamos la red neuronal
    print("Entrenando la red neuronal...")
    model.fit(x_train, y_train, epochs=10, batch_size=128)  # Entrenamos durante 10 épocas, usando paquetes de 128 imágenes

    # Evaluamos la red neuronal con los datos de prueba
    print("Evaluando la red neuronal...")
    loss, accuracy = model.evaluate(x_test, y_test)  # Calculamos la pérdida y la precisión en el conjunto de prueba
    print(f"Pérdida: {loss}, Precisión: {accuracy}")  # Mostramos los resultados

    plt.show() 