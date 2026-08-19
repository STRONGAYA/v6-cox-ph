# Alpine-based Docker image for v6-cox-ph algorithm
FROM python:3.10-alpine

# Install git (required for pip install from git)
RUN apk add --no-cache git

# Package name argument
ARG PKG_NAME="v6_cox_ph"

# Copy the entire repository
COPY . /app
WORKDIR /app

# Install the package
RUN pip install --no-cache-dir /app

# Set environment variable for package name
ENV PKG_NAME=${PKG_NAME}

# Command to run when container starts
CMD ["python", "-c", "from vantage6.algorithm.tools.wrap import wrap_algorithm; wrap_algorithm()"]
