FROM python:3.7
RUN pip install poetry
WORKDIR /turbot
COPY . ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-ansi --no-interaction
CMD ["poetry", "run", "turbot"]
