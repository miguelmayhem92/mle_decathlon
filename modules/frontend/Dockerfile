FROM python:3.11-slim

ARG root="modules"
ARG module="frontend"

WORKDIR /app

RUN pip install poetry

COPY ./${root}/${module}/pyproject.toml .

RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-root

COPY ./${root}/${module}/welcome.py .

EXPOSE 8501

CMD ["streamlit", "run", "welcome.py", "--server.port=8501", "--server.address=0.0.0.0"]