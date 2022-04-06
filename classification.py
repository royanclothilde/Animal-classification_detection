from PIL import Image
import os
import tensorflow as tf
from dattime import datetime

dictionary = {"araignee":0, "chat":1, "cheval":2, "chien":3, "ecureil":4,
          "elephant":5, "mouton":6, "papillon":7, "poule":8, "vache":9}


def list_folder(path):
    """ list all folser in a directionary"""
    list_folder = []
    for im in os.listdir(path):
        if im[0] != ".":
            list_folder.append(im)
    return list_folder


def list_image(path):
    """ list all image (.jpeg format) from a folder"""
    list_img = []
    for im in os.listdir(path):
        if im.split(".")[1] == "jpeg":
            list_img.append(im)
    return list_img


def get_image_details(path_img):
    """show size and mode of a particular image"""
    img = Image.open(path_img)
    print(img.size)
    print(img.mode)


def preprocess_images(path_img):
    """ resize image to 300x300 px"""
    img = Image.open(path_img)
    img_rezised = img.resize((300, 300))
    img_rezised.save(path_img)


batch_size = 32
img_height = 300
img_width = 300

# Create a train dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
  "image/",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Create a validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
  "image/",
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


class_names = train_ds.class_names
normalization_layer = tf.keras.layers.Rescaling(1./255)
num_classes = 10

# Create model
model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

# Train model
model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=5
)

# Save model
model.save('Models/model'+datetime.now().strftime("%Y-%m%d_%H:%M"))
# new_model = tf.keras.models.load_model('Models/')
