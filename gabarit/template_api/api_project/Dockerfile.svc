# base image
FROM python:3.8

LABEL maintainer="Agence Data Services"
LABEL description="Service REST {{package_name}}"

WORKDIR /app

# Install package
COPY .env .env
COPY pyproject.toml pyproject.toml
COPY setup.py setup.py
COPY launch.sh launch.sh
COPY {{package_name}}/ {{package_name}}/

RUN chmod +x /app/launch.sh && pip install --upgrade pip && pip install .

# Start API
EXPOSE 5000
CMD ["/app/launch.sh"]