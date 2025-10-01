# Spam checking using vector embeddings

Filtering spam mail is an arms race between hackers and security teams, but most spam emails in my inbox look really obviously to spot. Using vectors to represent the text content of the emails,
I found that even a really simple classifier could get an F1 score of 0.97 on a clean dataset.

As part of the [MLOps-zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) course, I've developed this into an end-to-end cloud pipeline that lets you simply type in some text and classify
whether or not it's spam. 

## Try it out!
[Link](https://amuveqngiu.us-east-1.awsapprunner.com/) (server may take some time to start up)

![MLOPs-spam-clasifier-demo](https://github.com/user-attachments/assets/73fd4bef-4ef6-45f7-8fee-6097120f4670)


## Architecture
<img width="2400" height="1148" alt="MLOps spam_ham architecture" src="https://github.com/user-attachments/assets/a660a559-7d8c-42ee-a1a8-4de40d811a58" />

## Installation instructions
Clone the repo and install python packages
```
git clone https://github.com/amorsi1/MLOps_spam_classifier
pip install pipenv
cd Embedded-spam-MLOps
pipenv install --dev
```

a `.env` file is used to centralize environmental variables, before running any code locally make sure to create this file and populate it with the following variables:
```.env
MLFLOW_TRACKING_URI=http://mlflow-server:8080 
MLFLOW_EXPERIMENT_NAME=spam-classifier
MLFLOW_MODEL_NAME=lr-model
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
```

With that set, you can download and preprocess the kaggle data, then run the training container, mlflow server container, and webapp container using docker-compose:
NOTE: data preprocessing will take 20-30 mins on an average laptop since it is front-loading all of the text embedding. Fortunately, you will only need to do this once.
```bash
make download_data
docker-compose up --build
```

or use the makefile to do the same (the makefile has additional testing capabilities)
```bash
make build
make up
```
## Dataset
A subset of the data in this [Phishing email dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) from Kaggle was used for model training. The 2 highest quality
datasets were combined and used.

## Testing
Unit testing and integration tests done using pytest. Note that the integration tests will fail if you don't have a docker daemon running







