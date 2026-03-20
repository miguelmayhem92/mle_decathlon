# build stage
FROM python:3.11-slim AS builder

ENV MODEL_DIR="dependencies_model"

ARG root="modules"
ARG module="inference"

WORKDIR /app

RUN pip install poetry

COPY pyproject.toml .

RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-root

COPY ./${root}/${module}/build.py .
RUN python build.py --experiment_name forecasting --run_name run_0

# run stage
FROM public.ecr.aws/lambda/python:3.11

ENV MODEL_DIR="dependencies_model"

ARG root="modules"
ARG module="inference"

COPY --from=builder usr/local/lib/python3.11/site-packages ${LAMBDA_RUNTIME_DIR}/

COPY --from=builder /app/dependencies_model/ ${LAMBDA_TASK_ROOT}/model/
COPY --from=builder /app/dep_features/ ${LAMBDA_TASK_ROOT}/dep_features/
COPY ./${root}/${module}/src ${LAMBDA_TASK_ROOT}/src/
COPY ./${root}/${module}/main.py ${LAMBDA_TASK_ROOT}

CMD [ "main.handler" ]