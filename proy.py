# -*- coding: utf-8 -*-
"""
@author: BriaN
"""

#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
np.random.seed(4321)

from sklearn.metrics import confusion_matrix
import itertools

import tensorflow as tf


from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
#Step 2 :Haciendo Diccionario de imágenes y etiquetas


# Fusionando imágenes de ambas carpetas HAM10000_images_part1 y HAM10000_images_part2 en un diccionario

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join( '*', '*.jpg'))}

# Este diccionario es útil para mostrar etiquetas más fáciles de usar más adelante.


lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

#Step 3 : Lectura y procesamiento de datos
skin_df = pd.read_csv(os.path.join( 'HAM10000_metadata.csv'))

# Creación de nuevas columnas para una mejor legibilidad

skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)
skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get) 
skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes
#Step 4 : Limpieza de datos
print(skin_df.isnull().sum())
skin_df['age'].fillna((skin_df['age'].mean()), inplace=True)
print(skin_df.isnull().sum())
print(skin_df.dtypes)
#Step 5 : EDA Análisis exploratorio de datos
fig, ax1 = plt.subplots(1, 1, figsize= (10, 5))
skin_df['cell_type'].value_counts().plot(kind='bar', ax=ax1)
print(skin_df['cell_type'].value_counts())
skin_df['dx_type'].value_counts().plot(kind='bar')
skin_df['localization'].value_counts().plot(kind='bar')
skin_df['age'].hist(bins=40)
skin_df['sex'].value_counts().plot(kind='bar')
sns.scatterplot('age','cell_type_idx',data=skin_df)
sns.catplot('sex','cell_type_idx',data=skin_df)
#Step 6: Carga y cambio de tamaño de imágenes
skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((125,100))))
print(skin_df.head())
n_samples = 5
fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
for n_axs, (type_name, type_rows) in zip(m_axs, 
                                         skin_df.sort_values(['cell_type']).groupby('cell_type')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=8475).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')
fig.savefig('category_samples.png', dpi=300)
print(skin_df['image'].map(lambda x: x.shape).value_counts())
features=skin_df.drop(columns=['cell_type_idx'],axis=1)
target=skin_df['cell_type_idx']
print(features.head())
#Step 7 : division de prueba de entrenamiento
x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(features, target, test_size=0.25,random_state=666)
print(tf.unique(x_train_o.cell_type.values))
sum = 0
for i in y_test_o:
    if i == 3:
        sum+=1
sum
print(sum)
#Step 8 : Normalización
x_train = np.asarray(x_train_o['image'].tolist())
x_test = np.asarray(x_test_o['image'].tolist())

x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)

x_test_mean = np.mean(x_test)
x_test_std = np.std(x_test)

x_train = (x_train - x_train_mean)/x_train_std
x_test = (x_test - x_test_mean)/x_test_std

#Step 9 : Codificación de etiquetas
# codificación de un solo uso en las etiquetas
y_train = to_categorical(y_train_o, num_classes = 7)
y_test = to_categorical(y_test_o, num_classes = 7)
print(y_test)
#Step 10 : splits de entrenamiento y validación
x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.1, random_state = 999)

# Remodelar la imagen en 3 dimensiones (altura a 100, anchura a 125 , canal a 3)
x_train = x_train.reshape(x_train.shape[0], *(100, 125, 3))
x_test = x_test.reshape(x_test.shape[0], *(100, 125, 3))
x_validate = x_validate.reshape(x_validate.shape[0], *(100, 125, 3))
#Step 11: Construcción del modelo
# CNN
#es una red neuronal artificial inspirada en los procesos biológicos en funcionamiento cuando las células nerviosas (neuronas) en el cerebro se conectan entre sí y responden a lo que el ojo ve. 
input_shape = (100, 125, 3)
num_classes = 7

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding = 'Same',input_shape=input_shape))
model.add(Conv2D(32,kernel_size=(3, 3), activation='relu',padding = 'Same',))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.16))

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding = 'Same'))
model.add(Conv2D(32,kernel_size=(3, 3), activation='relu',padding = 'Same',))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.20))

model.add(Conv2D(64, (3, 3), activation='relu',padding = 'same'))
model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
#Step 12: Configuración del Optimizador y Anneale
# Definir el optimizador
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# Compilar el modelo
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
# Establecer una tasa de aprendizaje annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=4, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
# Con aumento de datos para evitar el sobreajuste

datagen = ImageDataGenerator(
        featurewise_center=False,  # estableciendo la media de entrada en 0 sobre el dataset
        samplewise_center=False,  # estableciendo cada media de muestra en 0
        featurewise_std_normalization=False,  # dividiendo las entradas por variación standar del conjunto de datos
        samplewise_std_normalization=False,  # dividiendo cada entrada por su variación standar
        zca_whitening=False,  # APLICANDO ZCA whitening
        rotation_range=10,  # rotando aleatoriamente las imágenes en el rango (grados, 0 a 180)
        zoom_range = 0.1, # Imagen de zoom aleatorio
        width_shift_range=0.12,  # desplazando aleatoriamente las imágenes horizontalmente (fracción de ancho total)
        height_shift_range=0.12,  # desplazando aleatoriamente las imágenes verticalmente (fracción de la altura total)
        horizontal_flip=True,  # volteando imágenes al azar HORIZONTAL
        vertical_flip=True)  # volteando imágenes al azar VERTICAL

datagen.fit(x_train)
#Step 13: Montaje del modelo
# Ajustar el modelo
epochs = 60
batch_size = 16
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_validate,y_validate),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])
#Step 14: Evaluación del modelo
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
loss_v, accuracy_v = model.evaluate(x_validate, y_validate, verbose=1)
print("Validación: precisión = %f  ;  pérdida_v = %f" % (accuracy_v, loss_v))
print("Prueba: precisión = %f  ;  pérdida = %f" % (accuracy, loss))
model.save("model.h5")
# Función para trazar la matriz de confusión   
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Esta función imprime y traza la matriz de confusión.
    La normalización se puede aplicar estableciendo `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Etiqueta real')
    plt.xlabel('Etiqueta predecida')
    plt.show()
# Predeciendo los valores de validación del dataset
Y_pred = model.predict(x_validate)
# Convertiendo clases de predicciones en vectores
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convertiendo observaciones de validación en un vector
Y_true = np.argmax(y_validate,axis = 1) 
# calculando la matriz de confusión
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
plt.show()

# plot a la matriz de confusión
plot_confusion_matrix(confusion_mtx, classes = range(7)) 
plt.show()
# Predeciendo los valores de validación del dataset
Y_pred = model.predict(x_test)
# Convertiendo clases de predicciones en vectores
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convertiendo observaciones de validación en un vector
Y_true = np.argmax(y_test,axis = 1) 
# calculando la matriz de confusión
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

 

# plot a la matriz de confusión
plot_confusion_matrix(confusion_mtx, classes = range(7)) 
plt.show()
label_frac_error = 1 - np.diag(confusion_mtx) / np.sum(confusion_mtx, axis=1)
plt.bar(np.arange(7),label_frac_error)
plt.xlabel('Etiquetas')
plt.ylabel('Fracción clasificada incorrectamente')
# Función para trazar la pérdida de validación y la precisión de validación del modelo
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # resumiendo el historial para la precisión
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # resumiendo el historial de pérdidas
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
plot_model_history(history)
history.history.keys()    
