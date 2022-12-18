import tensorflow as tf

MODEL_FILE = "models/efficient_net_hidden256.hdf5"
SAVED_MODEL_PATH = "models/cat_breed_model"

model = tf.keras.models.load_model(MODEL_FILE)
tf.saved_model.save(model, SAVED_MODEL_PATH)