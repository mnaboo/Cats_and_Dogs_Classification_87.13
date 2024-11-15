import tensorflow as tf
from keras import Sequential
from keras.src.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Dropout, BatchNormalization, AveragePooling2D
import matplotlib.pyplot as plt
import cv2
import numpy as np


input_shape = (256, 256, 3)


# Wczytywanie danych treningowych i testowych
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory='./dogs_vs_cats/train',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256)
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    directory='./dogs_vs_cats/test',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256)
)

# Funkcja normalizująca obrazy
def process(image, label):
    image = tf.cast(image / 255., tf.float32)
    return image, label

train_ds = train_ds.map(process)
test_ds = test_ds.map(process)

# Tworzenie modelu CNN
model = Sequential()

model.add(Input(shape=input_shape))

model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
model.add(Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu'))

model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid'))

model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
model.add(Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu'))

model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid'))

model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
model.add(Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu'))

model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid'))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(2, activation='softmax'))


initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True
)
# Kompilacja modelu
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Trening modelu
history = model.fit(train_ds, epochs=10, validation_data=test_ds)

print("------")

def test_image(model, image_path):
    test_img = cv2.imread(image_path)
    test_img = cv2.resize(test_img, (256, 256))
    test_input = test_img.reshape((1, 256, 256, 3))
    prediction = model.predict(test_input)
    predicted_class = np.argmax(prediction)
    class_labels = ['cat', 'dog']
    predicted_label = class_labels[predicted_class]
    print(f'Prediction for {image_path}:', prediction)
    print('Predicted label:', predicted_label)
    print("------")

# Testowanie różnych obrazów
image_paths = ['cat.1.jpg', 'cat.2.jpg', 'cat.3.jpg', 'dog.1.jpg', 'dog.2.jpg', 'dog.3.jpg']
for image_path in image_paths:
    test_image(model, image_path)

# Wizualizacja wyników treningu
plt.plot(history.history['accuracy'], color='red', label='train')
plt.plot(history.history['val_accuracy'], color='blue', label='test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], color='red', label='train')
plt.plot(history.history['val_loss'], color='blue', label='test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()