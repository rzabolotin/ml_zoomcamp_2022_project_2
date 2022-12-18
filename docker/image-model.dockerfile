FROM tensorflow/serving:2.7.0
COPY models/cat_breed_model /models/cat_breed_model/1
ENV MODEL_NAME=cat_breed_model