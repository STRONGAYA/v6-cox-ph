ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-alpine@sha256:1b668429b3511ab407d8e00648891631b0b1a4d7e15e3ca70f38ab5b91ad4ab4 AS builder

# git: only needed by forks with git+https dependencies; never reaches runtime
RUN apk add --no-cache git

WORKDIR /src
COPY requirements-lock.txt pyproject.toml ./
COPY v6-cox-ph ./v6-cox-ph
# STRONG AYA: the two git dependencies are direct URLs (tag-pinned), which
# pip cannot unify with the lock's SHA-normalised direct-URL lines. Install
# the locked PyPI pins, then the git deps at their tags, then the package.
RUN python -m venv /opt/venv \
 && grep -v ' @ git+' requirements-lock.txt > /tmp/constraints.txt \
 && /opt/venv/bin/pip install --no-cache-dir -r /tmp/constraints.txt \
 && /opt/venv/bin/pip install --no-cache-dir \
      "vantage6-strongaya-general @ git+https://github.com/STRONGAYA/v6-tools-general.git@v1.0.5" \
      "vantage6-strongaya-rdf @ git+https://github.com/STRONGAYA/v6-tools-rdf.git@v1.1.0" \
 && /opt/venv/bin/pip install --no-cache-dir --no-deps .

FROM python:${PYTHON_VERSION}-alpine@sha256:1b668429b3511ab407d8e00648891631b0b1a4d7e15e3ca70f38ab5b91ad4ab4

ARG PKG_NAME="v6-cox-ph"
ENV PKG_NAME=${PKG_NAME} \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1

COPY --from=builder /opt/venv /opt/venv

CMD ["python", "-c", "from vantage6.algorithm.tools.wrap import wrap_algorithm; wrap_algorithm()"]
