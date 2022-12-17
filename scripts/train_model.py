from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB4
from PIL import ImageFile
from loguru import logger

ImageFile.LOAD_TRUNCATED_IMAGES = True

CAT_IMAGE_PATH = "data/cats"
MODEL_PATH = "models/efficient_net_hidden256.hdf5"


def get_settings():
    IMG_SIZE = 150
    IMG_CHANNELS = 3

    settings = {
        "EPOCHS": 5,
        "BATCH_SIZE": 98,
        "LR": 1e-4,
        "VAL_SPLIT": 0.15,
        "RANDOM_SEED": 42,
        "CLASS_NUM": 68,
        "IMG_SIZE": IMG_SIZE,
        "IMG_CHANNELS": IMG_CHANNELS,
        "INPUT_SHAPE": (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
    }

    return settings


def get_datagen(settings):
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        validation_split=settings.VAL_SPLIT,
        vertical_flip=True)

    train_generator = train_datagen.flow_from_directory(
        directory=CAT_IMAGE_PATH,
        target_size=(settings.IMG_SIZE, settings.IMG_SIZE),
        batch_size=settings.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=settings.RANDOM_SEED,
        subset='training')  # set as training data

    valid_generator = train_datagen.flow_from_directory(
        directory=CAT_IMAGE_PATH,
        target_size=(settings.IMG_SIZE, settings.IMG_SIZE),
        batch_size=settings.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=settings.RANDOM_SEED,
        subset='validation')  # set as training data

    return train_generator, valid_generator


def get_model(settings):
    model = keras.Sequential()

    model.add(EfficientNetB4(weights='imagenet', include_top=False, input_shape=settings.INPUT_SHAPE))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(settings.CLASS_NUM, activation='softmax'))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.Adam(learning_rate=settings.LR),
        metrics=["accuracy", "top_k_categorical_accuracy"]
    )

    return model


def train_model():
    settings = get_settings()
    train_generator, valid_generator = get_datagen(settings)
    model = get_model(settings)

    model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=valid_generator,
        validation_steps=len(valid_generator),
        epochs=settings.EPOCHS
    )

    return model


def main():
    logger.info("Start training model")
    model = train_model()
    logger.info(f"Saving model to {MODEL_PATH}")
    model.save(MODEL_PATH)
    logger.success("Done")

    model.save("models/model3_wout_dropout.hdf5")


if __name__ == "__main__":
    main()
