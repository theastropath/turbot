FROM python:3.8 as builder

COPY . /turbot

RUN pip wheel --wheel-dir python-wheels /turbot

FROM python:3.8-slim

COPY --from=builder python-wheels /python-wheels
RUN pip install --no-index --find-links /python-wheels /python-wheels/* && rm -rf /python-wheels

CMD ["turbot"]
