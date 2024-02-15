# base image
FROM python:3.8-bookworm AS builder

LABEL org.opencontainers.image.author="Agence Data Services"
LABEL org.opencontainers.image.description="Service REST {{package_name}}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=true

RUN python -m venv /opt/venv \
    && pip install --upgrade pip
ENV VIRTUAL_ENV="/opt/venv" PATH="/opt/venv/bin:${PATH}"

WORKDIR /usr/src/app
COPY pyproject.toml /usr/src/app/
RUN pip install .

FROM python:3.8-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends libmimalloc2.0 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/usr/src/app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /usr/src/app
COPY .env launch.sh /usr/src/app/
COPY {{package_name}}/ {{package_name}}/

RUN chmod +x launch.sh

# Start API
EXPOSE 5000
CMD ["/usr/src/app/launch.sh"]
