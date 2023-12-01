# Clasificador de Dígitos con Interfaz Gráfica

Este programa utiliza una red neuronal convolucional para clasificar dígitos escritos a mano del conjunto de datos MNIST. 
La interfaz gráfica permite a los usuarios seleccionar una imagen y ver la predicción realizada por la red neuronal.

## Requisitos

- Python 3.x
- Bibliotecas Python: tensorflow, numpy, Pillow

Puedes instalar las bibliotecas necesarias ejecutando el siguiente comando:

```bash
pip install tensorflow numpy Pillow

si quieres modificar el aprendizaje de la red neuronal solo debes modificar durante cuantas iteraciones quieres que se entrene con los datos
esto lo puedes encontrar en la linea 38 en el apartado epochs=10, por recomendacion lo dejamos en 10 ya que tiene una presicion del 99%, 
pero puedes bajar hasta un minimo de 5 iteracione y sera considerable la presicion.