# build stage
FROM public.ecr.aws/lambda/python:3.11 AS builder

ENV MODEL_DIR="dependencies_model"
ARG root="modules"
ARG module="inference"

WORKDIR /app

RUN pip install poetry

COPY ./${root}/${module}/pyproject.toml .

RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-root

COPY ./${root}/${module}/build.py .
RUN python build_model.py --experiment_name my_experiment --run_name run_1

# run stage
FROM public.ecr.aws/lambda/python:3.11

ENV MODEL_DIR="dependencies_model"
ARG root="modules"
ARG module="inference"

COPY --from=builder app/asset-layer/modules ${LAMBDA_RUNTIME_DIR}

COPY --from=builder /app/dependencies_model/ ${LAMBDA_TASK_ROOT}/model/
COPY --from=builder /app/dep_features/ ${LAMBDA_TASK_ROOT}/dep_features/
COPY ./${root}/${task_name}/src ${LAMBDA_TASK_ROOT}/src/
COPY ./${root}/${task_name}/main.py ${LAMBDA_TASK_ROOT}

CMD [ "main.handler" ]