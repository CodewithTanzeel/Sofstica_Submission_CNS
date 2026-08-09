# ------------------------------------------------------------
# 1️⃣  Base image – slim Python with OS utilities
# ------------------------------------------------------------
FROM python:3.13-slim-bullseye AS base

# Install OS‑level dependencies that some packages need
# (gcc & libgomp for XGBoost, plus curl for health‑checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non‑root user (Render runs containers as root by default,
# but using a non‑root user is a good practice)
ARG USERNAME=appuser
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ${USERNAME} && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME}

# ------------------------------------------------------------
# 2️⃣  Set workdir and copy only the files needed for
#     installing dependencies (this caches the pip layer)
# ------------------------------------------------------------
WORKDIR /app

# Copy the lockfile / requirements first – this is the layer that
# changes least often, so subsequent builds are fast.
COPY requirements.txt ./
# If you have a `requirements-lock.txt` or `pipfile.lock` you can
# copy that instead and run `pip install -r requirements-lock.txt`.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------
# 3️⃣  Copy the rest of the source code
# ------------------------------------------------------------
COPY . ./

# ------------------------------------------------------------
# 4️⃣  Switch to the non‑root user
# ------------------------------------------------------------
USER ${USERNAME}

# ------------------------------------------------------------
# 5️⃣  Expose the port Render will set (it injects $PORT)
# ------------------------------------------------------------
EXPOSE 8080   # Render forwards $PORT → this internal port

# ------------------------------------------------------------
# 6️⃣  Runtime command
# ------------------------------------------------------------
# Render sets the environment variable PORT at runtime.
# gunicorn reads it automatically when we bind to 0.0.0.0:$PORT.
#
# The dash app exposes a Flask `server` object, so we point
# gunicorn at that WSGI entry point.
#
#   gunicorn dash_app:server --bind 0.0.0.0:$PORT
#
# Adding `--workers 2` gives a tiny amount of concurrency
# (still within the free tier limits). Feel free to adjust.
#
# NOTE: The `--timeout 120` prevents gunicorn from killing the
# process during a long first‑time model load.
CMD ["gunicorn", "dash_app:server", "--bind", "0.0.0.0:$PORT", "--workers", "2", "--timeout", "120"]
