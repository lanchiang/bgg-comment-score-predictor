# document-text-extraction-api-tmp

## Contact

Melange development team <Melange development team@stepstone.com>

## Setup

```shell
poetry install --with dev
```

## Running the application locally

```shell
uvicorn app.main:setup --host localhost --factory --reload
```

### With Docker

```shell
docker build -t document-text-extraction-api-tmp --target runtime --load .

docker run -it -p 80:80 -e OTEL_AGENT_HOST="0.0.0.0" -e OTEL_SERVICE=document-text-extraction-api-tmp -e OTEL_VERSION=0.0.0 -e OTEL_ENV=exp document-text-extraction-api-tmp
```

### Running build docker locally

```shell
docker build -t document-text-extraction-api-tmp-build --target build --load .

docker run -it -p 80:80 document-text-extraction-api-tmp
```

## Generating openapi specification automatically

```shell
cd deployment
poetry run python -m scripts.generate_spec
```
