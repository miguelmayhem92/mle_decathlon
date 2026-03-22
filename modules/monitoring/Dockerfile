FROM python:3.11-slim

ARG root="modules"
ARG data_folder="data"
ARG train_folder="monitoring"

WORKDIR /app

RUN pip install poetry

COPY ./${root}/${train_folder}/pyproject.toml .

RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-root

COPY ./${root}/${data_folder} ./${data_folder}
COPY ./${root}/${train_folder}/src_monitor ./src_monitor
COPY ./${root}/${train_folder}/main.py .

CMD ["python", "main.py"]