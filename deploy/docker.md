# Docker (local) - optional

Minimal example:
```bash
docker build -t heartcloud .
docker run -p 8000:8000 --env-file .env heartcloud
```
