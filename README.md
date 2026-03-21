
archi overview:

![alt text](https://github.com/miguelmayhem92/mle_decathlon/blob/main/images/mle_decathlon.drawio.png)

how to run?

make sure you have docker in your machine and you are running on linux (otherwise some endpoints would crash if running on windows)

first run mlflow server with:

```
docker compose up -d mlflow_server
```

then train the model:

```
docker compose up -d trainer
```

then deploy the inference service

```
docker compose up -d inference
```

finally the ui

```
docker compose up -d frontend
```