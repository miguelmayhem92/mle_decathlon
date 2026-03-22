# build stage
FROM python:3.11-slim AS builder

ENV MODEL_DIR="dependencies_model"

ARG root="modules"
ARG module="inference"

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


COPY ./${root}/${module}/build.py .
RUN python build.py --experiment_name forecasting --run_name run_0

# run stage
FROM public.ecr.aws/lambda/python:3.11 AS run

ENV MODEL_DIR="dependencies_model"

ARG root="modules"
ARG module="inference"

# COPY --from=builder usr/local/lib/python3.11/site-packages ${LAMBDA_RUNTIME_DIR}/
COPY --from=builder /app/deps/ ${LAMBDA_RUNTIME_DIR}/

COPY --from=builder /app/dependencies_model/ ${LAMBDA_TASK_ROOT}/model/
COPY --from=builder /app/dep_features/ ${LAMBDA_TASK_ROOT}/dep_features/
COPY ./${root}/${module}/main.py ${LAMBDA_TASK_ROOT}

CMD [ "main.handler" ]