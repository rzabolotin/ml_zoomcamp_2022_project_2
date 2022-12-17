# Project: cat's breed detection
Project created at cohort 2022 of ML Zoomcamp course.

The solved problem is a classification problem. We try to predict the breed of a cat by its photo.  
It can be useful for cat's owners to know the breed of their cat, and for cat's shelters to know the breed of the cat to find the owner.  
I created telegram bot for this project. That's a user interface for the model. You can find it [here](https://t.me/cat_breed_detection_bot).  

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

The best result was achieved with EfficientNetB4  + hidden(256).  

I also tried to use different optimizers: Adam, SGD, RMSprop. The best result was achieved with Adam.

The resulted model is [published in Google Drive](https://drive.google.com/file/d/1CtAl6MsrqnWzLDMjWiYJasNjSuT2MsLG/view?usp=sharing)  
Script for training the model is [here](/scripts/train_model.py)  

If you want to train the model, you need to download and unzip the [dataset](https://drive.google.com/file/d/1Csr2tC8SZDd___rIibFnI58sXaSkjHMr/view?usp=share_link)   to the folder `data`, and then run the script.

# Used technologies

- Python
- Tensorflow
- Saturn Cloud (https://www.saturncloud.io/)
- // Streamlit (https://www.streamlit.io/)
