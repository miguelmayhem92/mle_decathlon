FROM python:3.11-slim

ARG root="modules"
ARG data_folder="data"
ARG train_folder="model_training"

WORKDIR /app

RUN pip install poetry

COPY pyproject.toml .

RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-root

COPY ./${root}/${data_folder} ./${data_folder}
COPY ./${root}/${train_folder}/src ./src
COPY ./${root}/${train_folder}/configs ./configs
COPY ./${root}/${train_folder}/train.py .

# CMD ["python", "train.py"]
RUN python train.py