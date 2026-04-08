# Face Composite

Face swap tool for casting pitch decks. Composites actor headshots into scene/mood images using InsightFace + GFPGAN.

## Quick start

```bash
docker compose up --build
```

First startup downloads ~1.5 GB of ML models into the Docker volume.

- **Frontend:** http://localhost:3030
- **Backend API:** http://localhost:8030

## How to use

1. Drop a scene image (left panel) — faces are auto-detected
2. Click a face in the scene to select the swap target
3. Drop an actor headshot (right panel)
4. Click **Swap Face**
5. Preview with before/after slider, then download PNG

## Architecture

- **Backend:** FastAPI + InsightFace inswapper_128 + GFPGAN face restoration
- **Frontend:** Next.js 15 + Tailwind CSS
- **Runtime:** Docker Compose, runs fully offline after initial model download

## API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/detect-faces` | POST | Detect faces, returns bounding boxes |
| `/swap` | POST | Swap face in scene with actor face |
| `/enhance` | POST | Standalone face enhancement |

## Running without Docker

### Backend
```bash
cd backend
pip install -r requirements.txt
MODEL_DIR=./models bash download_models.sh
MODEL_DIR=./models uvicorn main:app --port 8030
```

### Frontend
```bash
cd frontend
npm install
INTERNAL_API_URL=http://localhost:8030 npm run dev -- -p 3030
```
