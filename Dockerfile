# -----------------------------
# Python environment for:
# GANs_for_Network_Intrusion_Data
#
# Build from the folder containing this Dockerfile:
#
#   docker build --progress=plain -t gan-env .
#
# Run the container interactively:
#
#   docker run --rm -it gan-env
#
# Verify the installed packages inside the container:
#
#   python -c "import numpy, pandas, matplotlib, keras, tensorflow, seaborn; print('ok')"
#
# Enter the NSL-KDD model directory:
#
#   cd NSL-KDD
#
# Train and test the model:
#
#   python train.py
#   python test.py
#
# Example:
#
#   cd ../UNSW-NB15
#   python train.py
#   python test.py
# -----------------------------

FROM python:3.7-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Git and system dependencies required by older Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        gcc \
        g++ \
        libopenblas-dev \
        liblapack-dev \
        libfreetype6-dev \
        libpng-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Clone the project repository
RUN git clone --depth 1 \
    https://github.com/mayara-santos01/GANs_for_Network_Intrusion_Data.git

# Python 3.7 is not compatible with current pip releases.
# Use versions compatible with Python 3.7.
RUN python -m pip install \
        --upgrade \
        "pip==23.2.1" \
        "setuptools==65.7.0" \
        "wheel==0.38.4"

# Install the project dependencies.
# The increased timeout and retry settings help prevent PyPI download failures.
RUN python -m pip install \
        --timeout 300 \
        --retries 10 \
        numpy==1.19.5 \
        pandas==1.2.5 \
        matplotlib==3.4.3 \
        keras==2.2.4 \
        tensorflow==2.4.4 \
        pandas-profiling==3.1.0 \
        tabulate==0.8.10 \
        imbalanced-learn==0.8.0 \
        category_encoders==2.2.2 \
        scikit-learn==0.24.2 \
        seaborn==0.11.2

# Set the default directory when the container starts
WORKDIR /app/GANs_for_Network_Intrusion_Data

# Start an interactive shell
CMD ["bash"]
