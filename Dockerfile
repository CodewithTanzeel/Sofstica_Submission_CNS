# ---------- Base image ----------
FROM python:3.13-slim

# ---------- Working directory ----------
WORKDIR /app

# ---------- Install system deps ----------
# (optional – often needed for scientific libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

# ---------- Copy project files ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code (including dash_app.py, src/, results/ etc.)
COPY . .

# ---------- Expose the port ----------
EXPOSE 8050

# ---------- Runtime command ----------
# Uses the Procfile command under the hood
CMD ["gunicorn", "dash_app:server", "--bind", "0.0.0.0:8050", "--workers", "2"]
