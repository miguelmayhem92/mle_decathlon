FROM python:3.11-slim


ARG root="modules"
ARG data_folder="data"
ARG train_folder="model_training"
ARG common="common"

ENV MODULE_PATH="${root}/${train_folder}"
ENV COMMON_PATH="${root}/common"
ENV DATA_PATH="${data_folder}"
ENV DATA_FOLDER="data"

WORKDIR /app/

RUN pip install poetry

COPY pyproject.toml .

RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-root

COPY ./${root}/${data_folder} ./${DATA_PATH}
COPY ./${root}/${train_folder}/src ./${MODULE_PATH}/src
COPY ./${root}/${train_folder}/configs ./${MODULE_PATH}/configs
COPY ./${root}/${train_folder}/train.py ./train.py 
COPY ./${root}/${common} ./${COMMON_PATH}

CMD ["sh", "-c", "python train.py"]
# RUN python train.py