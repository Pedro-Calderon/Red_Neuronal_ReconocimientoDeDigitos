import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tkinter import Tk, Button, Label, Canvas, filedialog
from PIL import ImageTk

# Cargar y preprocesar datos
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Construir el modelo de red neuronala
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compilar el modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Aumento de datos
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)
datagen.fit(train_images)

# Entrenar el modelo con aumento de datos
model.fit(datagen.flow(train_images, train_labels, batch_size=64), epochs=10, validation_data=(test_images, test_labels))

# Crear la interfaz gráfica
root = Tk()
root.title("Clasificador de Dígitos")

# Calcular las dimensiones y posición central de la pantalla
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_width = 400 
window_height = 400  
x_position = (screen_width - window_width) // 2
y_position = (screen_height - window_height) // 2

# Establecer las dimensiones y posición de la ventana
root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

# Función para cargar y preprocesar la imagen seleccionada
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Función para realizar la predicción y actualizar la etiqueta en la interfaz
def predict_digit():
    file_path = filedialog.askopenfilename()
    if file_path:
        img_array = load_and_preprocess_image(file_path)
        predictions = model.predict(img_array)
        predicted_label = np.argmax(predictions[0])
        result_label.config(text=f'La red predice que la imagen contiene el dígito: {predicted_label}')

        # Mostrar la imagen en el lienzo
        img = image.load_img(file_path, target_size=(300, 350), color_mode='grayscale')
        img = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor='nw', image=img)
        canvas.image = img  # Para evitar que la imagen se elimine por el recolector de basura

# Crear y configurar la etiqueta para mostrar el resultado
result_label = Label(root, text="")
result_label.pack()

# Crear y configurar el lienzo para mostrar la imagen seleccionada
canvas = Canvas(root, width=300, height=350)
canvas.pack()

# Crear y configurar el botón para seleccionar una imagen
button = Button(root, text="Seleccionar Imagen", command=predict_digit)
button.pack()

# Ejecutar la interfaz gráfica
root.mainloop()
