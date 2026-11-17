FROM python:3.11-slim
WORKDIR /app
# rlnav/ is copied in by scripts/sync_harbor.sh so the build context is self-contained.
COPY rlnav /opt/rlnav/rlnav
COPY pyproject.toml /opt/rlnav/pyproject.toml
RUN pip install --no-cache-dir /opt/rlnav
