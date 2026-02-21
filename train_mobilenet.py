import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

# Data augmentation for training set and rescaling for both training and validation sets

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 20,
    zoom_range = 0.2,
    horizontal_flip = True
)

val_datagen = ImageDataGenerator(rescale = 1./255)

train_gen = train_datagen.flow_from_directory(
    "data/train",
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = "categorical"
)

val_gen = val_datagen.flow_from_directory(
    "data/val",
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = "categorical"
)

num_classes = train_gen.num_classes

# MobileNetV2 model with transfer learning

base_model = MobileNetV2(
    input_shape = (224, 224, 3),
    include_top = False,
    weights = "imagenet"
)

base_model.trainable = False #Freeze base model layers

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation = "relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation = "softmax")
])

model.compile(
    optimizer = optimizers.Adam(learning_rate=1e-4),
    loss = "categorical_crossentropy",
    metrics = ["accuracy"]
)

# Callbacks

callbacks = [
    EarlyStopping(patience = 2, restore_best_weights = True),
    ModelCheckpoint("models/mobilenetv2_best.h5", save_best_only = True)
]

model.fit(
    train_gen,
    epochs = EPOCHS,
    validation_data = val_gen,
    callbacks = callbacks
)