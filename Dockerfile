FROM python:3.10-alpine

ARG PKG_NAME="v6-cox-ph"

RUN apk add --no-cache git

COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir /app

ENV PKG_NAME=${PKG_NAME}

CMD ["python", "-c", "from vantage6.algorithm.tools.wrap import wrap_algorithm; wrap_algorithm()"]
