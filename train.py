import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Data augmentation for training set and rescaling for both training and validation sets

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 20,
    zoom_range = 0.2,
    horizontal_flip = True,
)

val_datagen = ImageDataGenerator(rescale = 1./255)

train_gen = train_datagen.flow_from_directory(
    "data/train",
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = "categorical",
)

val_gen = val_datagen.flow_from_directory(
    "data/val",
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = "categorical",
)

num_classes = train_gen.num_classes

# CNN model from scratch

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (224, 224, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation = "relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation = "relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation = "relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation = "softmax"),
])

model.compile(
    optimizer = optimizers.Adam(learning_rate=1e-4),
    loss = "categorical_crossentropy",
    metrics = ["accuracy"],
)

model.summary()

# Callbacks for early stopping and model checkpointing

callbacks = [
    EarlyStopping(patience = 3, restore_best_weights = True),
    ModelCheckpoint("models/cnn_from_scratch.h5", save_best_only = True),
]

# Train

history = model.fit(
    train_gen,
    epochs = EPOCHS,
    validation_data = val_gen,
    callbacks = callbacks,
)

# Save final model
model.save("models/cnn_from_scratch_final.h5")