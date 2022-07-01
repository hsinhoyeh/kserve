#FROM python:3.7
FROM footprintai/aixserver:v0.5.1-base

COPY . .
ENTRYPOINT ["python", "-m", "aixserver"]
