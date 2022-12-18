import numpy as np
from urllib import request
from io import BytesIO
from PIL import Image
import grpc
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from proto import np_to_protobuf
import os

from flask import Flask, request as flask_request, jsonify

app = Flask("gateway")

MODEL_HOST = os.getenv("BREED_MODEL_URL", "localhost:8500")
IMAGE_SIZE = (150, 150)

channel = grpc.insecure_channel(MODEL_HOST)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

class_info = {'Abyssinian': 0,
              'American Bobtail': 1,
              'American Curl': 2,
              'American Shorthair': 3,
              'American Wirehair': 4,
              'Applehead Siamese': 5,
              'Balinese': 6,
              'Bengal': 7,
              'Birman': 8,
              'Bombay': 9,
              'British Shorthair': 10,
              'Burmese': 11,
              'Burmilla': 12,
              'Calico': 13,
              'Canadian Hairless': 14,
              'Chartreux': 15,
              'Chausie': 16,
              'Chinchilla': 17,
              'Cornish Rex': 18,
              'Cymric': 19,
              'Devon Rex': 20,
              'Dilute Calico': 21,
              'Dilute Tortoiseshell': 22,
              'Domestic Long Hair': 23,
              'Domestic Medium Hair': 24,
              'Domestic Short Hair': 25,
              'Egyptian Mau': 26,
              'Exotic Shorthair': 27,
              'Extra-Toes Cat - Hemingway Polydactyl': 28,
              'Havana': 29,
              'Himalayan': 30,
              'Japanese Bobtail': 31,
              'Javanese': 32,
              'Korat': 33,
              'LaPerm': 34,
              'Maine Coon': 35,
              'Manx': 36,
              'Munchkin': 37,
              'Nebelung': 38,
              'No cat': 39,
              'Norwegian Forest Cat': 40,
              'Ocicat': 41,
              'Oriental Long Hair': 42,
              'Oriental Short Hair': 43,
              'Oriental Tabby': 44,
              'Persian': 45,
              'Pixiebob': 46,
              'Ragamuffin': 47,
              'Ragdoll': 48,
              'Russian Blue': 49,
              'Scottish Fold': 50,
              'Selkirk Rex': 51,
              'Siamese': 52,
              'Siberian': 53,
              'Silver': 54,
              'Singapura': 55,
              'Snowshoe': 56,
              'Somali': 57,
              'Sphynx - Hairless Cat': 58,
              'Tabby': 59,
              'Tiger': 60,
              'Tonkinese': 61,
              'Torbie': 62,
              'Tortoiseshell': 63,
              'Turkish Angora': 64,
              'Turkish Van': 65,
              'Tuxedo': 66,
              'York Chocolate': 67}


def get_image_from_url(url, target_size):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    img = img.resize(target_size, Image.NEAREST)
    x = np.array(img, dtype='float32')
    batch = np.expand_dims(x, axis=0)
    return batch


def get_predictions(url):
    X = get_image_from_url(url, IMAGE_SIZE)

    pb_request = predict_pb2.PredictRequest()
    pb_request.model_spec.name = "cat_breed_model"
    pb_request.model_spec.signature_name = "serving_default"
    pb_request.inputs["efficientnetb4_input"].CopyFrom(np_to_protobuf(X))

    pb_response = stub.Predict(pb_request, timeout=20.0)

    predictions = pb_response.outputs['dense_17'].float_val

    class_predictions = dict(zip(class_info.keys(), predictions))
    return sorted(class_predictions.items(), key=lambda x: x[1], reverse=True)


@app.route("/predict", methods=["POST"])
def predict():
    url = flask_request.json["url"]
    predictions = get_predictions(url)
    return jsonify(predictions)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
