# Decathlon forecasting  and model deployment usecase


## Overview



## Architecture overview:

this MVP has the following architecture

![alt text](https://github.com/miguelmayhem92/mle_decathlon/blob/main/images/mle_decathlon.drawio.png)

reason of this architecure:

* intensive use of docker: all services are build using docker and orchestrated using docker compose. We look to build services that can be reproducible and portable in any environment
* intensive user of python: all services are mainly build using open-source python tooling
* service are build in modular manner: funcion, classes and services are responsible of one task at the time
* training job reproduces the datascientis model and underlaying buisness logic. Also it allows later code reproducibility in case retraining automation
* model repository allows to store model artifacts and dependencies therefore no other complicate solutions would be needed  (managed libs or non versioned experiments)
* inference service will deploy in its image the model and dependencies so that model wont depend on other services. Also it allows to comunicate with other services like external endpoints, or dedicated clients
* the frontend is an user interface that allows to interact with the model

## Repo overview

the repo architecture is the following:

```
.
├── docker-compose.yml
├── images
├── modules
│   ├── data
│   ├── eda
│   ├── frontend
│   ├── inference
│   ├── model_repository
│   └── model_training
│   └── monitoring
├── poetry.lock
├── pyproject.toml
└── README.md
```

services are separated in modules
* data -> It is repo dedicated to data files (in real prd escenario this should be replaced with feature store or other services that make extraction/ingestion)
* eda -> space dedicated to the datascients where he will research model and business logic
* model_repository -> it has the image code for model registry service (in prd scenario this folder is a dedicated project)
* model_training -> this folder have the model logic and code so that an image can train it
* inference -> folder dedicated to the inference code
* frontend -> folder of the user interface that will interact with the inference service
* Monitoring -> folder of the monitoring service using nannyml an on demand tasks

## How to run?

make sure you have docker in your machine and you are running on linux (otherwise some endpoints would crash if running on windows)

first run mlflow server with:

```
docker compose up -d mlflow_server
```

connect to the mlflow ui with the folling url

```
http://localhost:5000/
```

then train the model:

```
docker compose up -d trainer
```

the trainer will launch a training job that takes 30 seconds, so lets give it some time before running the next service

Once the model is trainned and logged to the remote registry, we can start building the inference service. This service will use aws Lambda image

then deploy the inference service

```
docker compose up -d inference
```

finally the ui built on streamlit

```
docker compose up -d frontend
```

to connect to the ui with the following url:

```
http://localhost:8501/
```

launch one monitor job

```
docker compose up -d monitoring
```