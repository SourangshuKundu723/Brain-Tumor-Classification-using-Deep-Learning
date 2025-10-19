import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_classification_generators(data_dir, img_size=(224,224), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_gen = train_datagen.flow_from_directory(
        f"{data_dir}/train",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training"
    )

    val_gen = train_datagen.flow_from_directory(
        f"{data_dir}/train",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation"
    )

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        f"{data_dir}/test",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    return train_gen, val_gen, test_gen
