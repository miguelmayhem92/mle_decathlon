# build stage
FROM python:3.11-slim AS builder

ARG root="modules"
ARG module="inference"
ARG module_training="model_training"
ARG MLFLOW_TRACKING_URI

ENV MODULE_PATH="${root}/${module}"
ENV MODEL_DEPS_PATH="${root}/${module_training}"

WORKDIR /app

RUN pip install poetry poetry-plugin-export

COPY ./${root}/${module}/pyproject.toml .

RUN poetry export -f requirements.txt --only main --without-hashes -o requirements.txt

RUN pip install numpy==1.26.*
# Install heavy packages as manylinux2014 pre-built wheels into /app/deps
RUN pip install --only-binary=:all: \
    --platform manylinux2014_x86_64 \
    --target /app/deps \
    --python-version 3.11 \
    --implementation cp \
    --upgrade \
    -r requirements.txt


# RUN mkdir -p ./${MODEL_DEPS_PATH}/src/custom_transformers/

COPY ./${root}/${module}/src_inference/build.py .

RUN python build.py --experiment_name forecasting --run_name run_0 --mlflow_url ${MLFLOW_TRACKING_URI}

# run stage
FROM public.ecr.aws/lambda/python:3.11 AS run

ARG root="modules"
ARG module="inference"

ENV DOWNLOAD_DIR="modules/model_training"
ENV MODULE_PATH="${root}/${module}"

WORKDIR ${LAMBDA_TASK_ROOT}

COPY --from=builder /app/deps/ ./
COPY --from=builder /app/${DOWNLOAD_DIR}/ ./${DOWNLOAD_DIR}/
COPY --from=builder /app/dep_features/ ./dep_features/

COPY ./${MODULE_PATH}/src_inference/inference.py ./${MODULE_PATH}/src_inference/
COPY ./${MODULE_PATH}/src_inference/models/ ./${MODULE_PATH}/src_inference/models/
COPY ./${MODULE_PATH}/main.py ${LAMBDA_TASK_ROOT}

CMD [ "main.handler" ]