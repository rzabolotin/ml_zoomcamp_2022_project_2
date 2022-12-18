# Project: cat's breed detection
Project created at cohort 2022 of ML Zoomcamp course.

The solved problem is a classification problem. We try to predict the breed of a cat by its photo.  
It can be useful for cat's owners to know the breed of their cat, and for cat's shelters to know the breed of the cat to find the owner.  

![image](/static/example_breeds.png)

# Sources of data
In this project, I used the data from the [Cat Breeds Dataset](https://www.kaggle.com/datasets/ma7555/cat-breeds-dataset) on Kaggle.
It contains 126'607 images of cats of 67 breeds.
The most popular breeds are:
- Domestic Short Hair: 53027
- Domestic Medium Hair: 5482
- American Shorthair: 5295
- Domestic Long Hair: 4499
- Persian: 4018
- Tortoiseshell: 3963
- Calico: 3468
- Torbie: 3396
- Dilute Calico: 3230
- Tuxedo: 3181
- Dilute Tortoiseshell: 3152
- Tabby: 3012
- Siamese: 2888
- Ragdoll: 2669
- Bengal: 2477
- Tiger: 2256

# Images with no cats
I also decided to add "No cat" category, to allow model find images where there is no cat.    
I used photos from [House Rooms Image Dataset](https://www.kaggle.com/datasets/robinreni/house-rooms-image-dataset). It contains 5'250 photos of rooms.  


# Preparing data

There is way too many photos of cats. To remove unbalance of domestic cats, and to make training faster, I decided to limit the number of photos for each breed to 1000.
  
The script for combining datasets and shrinking of breeds is [presented here](/scripts/prepare_dataset.py).  
The resulted dataset is [published in Google Drive](https://drive.google.com/file/d/1Csr2tC8SZDd___rIibFnI58sXaSkjHMr/view?usp=share_link)  

To run this project you need to download and unzip this dataset to the folder `data`.

# EDA
Let's look at some photos from the dataset.  
There is notebook with overview of the dataset [here](/notebooks/EDA.ipynb)

# Model selection
I decided to use transfer learning. I used keras framework and tensorflow backend.  
I have 68 classes in the dataset, so I decided to top 5 accuracy metric, because it is more suitable for multiclass classification.  

In this part of the project, [Saturn Cloud](https://www.saturncloud.io/) helped me a lot.  
With their Jupyter Lab server I ran [this notebook](/notebooks/model_selection.ipynb) to try different models.

I tried to use different models:
1. Xception + hidden(256) + dropout(0.25)
2. EfficientNetB4  + hidden(256) + dropout(0.25)
3. EfficientNetB4  + hidden(256)
4. EfficientNetB4  + hidden(100) + dropout(0.25)
5. EfficientNetB4  + hidden(100)

## Results
The best result was achieved with EfficientNetB4  + hidden(256).  

I also tried to use different optimizers: Adam, SGD, RMSprop. The best result was achieved with Adam.

The resulted model is [published in Google Drive](https://drive.google.com/file/d/1CtAl6MsrqnWzLDMjWiYJasNjSuT2MsLG/view?usp=sharing)  

Score for this model is 0.70 for top 5 accuracy metric.  
Script for training the model is [here](/scripts/train_model.py)  
If you want to train the model, you need to download and unzip the [dataset](https://drive.google.com/file/d/1Csr2tC8SZDd___rIibFnI58sXaSkjHMr/view?usp=share_link)   to the folder `data`, and then run the [script](scripts/train_model.py):

```shell
pipenv run python scripts/train_model.py
```

# Deployment
I used tensorflow saved_model format for deployment.  
To convert the model to `saved_model` format I used [convert_to_saved_model.py](/scripts/convert_to_saved_model.py)  
You need to train model first, or download model from [Google Drive](https://drive.google.com/file/d/1CtAl6MsrqnWzLDMjWiYJasNjSuT2MsLG/view?usp=sharing)

```shell
pipenv run python scripts/convert_to_saved_model.py
```

## Creating service
There is the [notebook](notebooks/tf_serving_connector.ipynb) with example of using deployed model in docker container.  
After that, I converted it to the [gateway.py](scripts/gateway.py) script.  
And then I created Flask app for the API.

I used [postman](https://www.postman.com/) for testing. To use the service you need to send POST request to the endpoint `http://localhost:9696/predict` with json body:
```json
{
  "url":"https://github.com/rzabolotin/ml_zoomcamp_2022_project_2/blob/main/static/burmila.jpg?raw=true"
}
```

## Containerization
I used docker and docker-compose for local deployment:

- [image-model](/docker/image-model.dockerfile) for building docker image for model serving.
- [image-gateway](/docker/image-gateway.dockerfile) for building docker image for flask gateway.
- [docker-compose.yml](docker-compose.yaml) for running docker containers together.

To run the project you need run docker-compose. It will build docker images and run containers.

```shell
docker-compose up
```

# Local kubernetes deployment

I used [kind](https://kind.sigs.k8s.io/) for local kubernetes deployment.

To run the project you need to run the following commands:
```shell
# create kubernetes cluster
kind create cluster 

# apply all kubernetes configs
kubectl apply -f kube-config/model-deployment.yaml 
kubectl apply -f kube-config/model-service.yaml
kubectl apply -f kube-config/gateway-deployment.yaml
kubectl apply -f kube-config/gateway-service.yaml

# make port forwarding to gateway service
kubectl port-forward service/gateway-service 80:9696
```

After that you can send the same  POST request to `http://localhost:9696/predict`, and service will reply with json answer.

# Deploying to AWS EKS
I used [eksctl](https://eksctl.io/) for creating EKS cluster.
```shell
eksctl create cluster -f kube-config/eks-config.yaml
```
Then you need to create ECR repository for docker images and push them there.
```shell
aws ecr create-repository --repository-name ml-zoomcamp
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com

docker tag breed_model:v3-001 123456789012.dkr.ecr.us-east-1.amazonaws.com/ml-zoomcamp:breed_model-v3-001
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/ml-zoomcamp:breed_model-v3-001

docker tag breed_gateway 123456789012.dkr.ecr.us-east-1.amazonaws.com/ml-zoomcamp:breed-gateway
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/ml-zoomcamp:breed-gateway
```

Then you need to apply kubernetes configs:
```shell
kubectl apply -f kube-config/model-deployment.yaml
kubectl apply -f kube-config/model-service.yaml
kubectl apply -f kube-config/gateway-deployment.yaml
kubectl apply -f kube-config/gateway-service.yaml
```

After that you can send the same  POST request, but sent it to EKS public API endpoint.  
(for me it was http://a1554c88daf744e1a85752b08be1e24c-1291281226.us-east-1.elb.amazonaws.com/predict, but I deleted the cluster, so it is not available now)

# Used technologies

- Python
- Tensorflow
- Saturn Cloud (https://www.saturncloud.io/)
- Docker
- Postman
- Kind
- AWS EKS