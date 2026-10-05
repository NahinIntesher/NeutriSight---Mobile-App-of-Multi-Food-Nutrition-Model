# NutriSight 1.1 — Professional integrated build

This package contains the Expo SDK 57 frontend, FastAPI backend, and all three supplied model checkpoints.

## What was fixed in 1.1

- Physical-phone networking no longer defaults to `localhost`. The Expo development PC address is detected automatically and port `8000` is used for the backend.
- Old saved `localhost` / `127.0.0.1` settings are automatically migrated on native devices.
- Login and signup are wired to `/api/auth/login` and `/api/auth/signup` with validation, secure token storage, connection checks, and useful network errors.
- Backend now has `/`, `/api`, `/api/ping`, `/health`, auth, profile, analysis, history, logout, and account deletion routes.
- Local-development CORS works from Expo web and LAN origins without cookie credentials.
- Welcome, login, signup, home, scan, result, journal, profile and settings were redesigned with a consistent professional UI.
- New NutriSight app icon, adaptive icon, splash logo, animated scan illustration, startup animation, and analysis loading animation.
- Camera/gallery -> `/api/analyze` -> YOLO -> ViT fallback -> Nutrition -> result -> history flow is connected.
- Settings includes backend health, model presence and an auto-detected server button.

## Model mapping

- `models/yolo/best.pt` — YOLO food segmentation
- `models/vit/best_model.pth` — ViT food fallback classifier
- `models/vit/classes.json` — classifier output order
- `models/nutrition/best_model.pth` — Nutrition5k ViT regression

## Run on Windows

### 1. Backend

Open `backend` and run:

```bat
start.bat
```

Keep that terminal open. It binds to `0.0.0.0:8000`, so your phone can reach it over Wi-Fi. The script prints the LAN URL. If Windows Firewall asks, allow Python on **Private networks**.

You can verify the backend in a browser:

```text
http://YOUR-PC-IP:8000/
http://YOUR-PC-IP:8000/health
http://YOUR-PC-IP:8000/docs
```

The root URL should return a NutriSight JSON status object, not `{"detail":"Not Found"}`.

### 2. Frontend

Open `frontend` in another terminal and run:

```bat
start.bat
```

or:

```powershell
npm install
npx expo-doctor
npx expo start --clear
```

Expo SDK 57 requires Node.js 22.13 or newer.

### 3. Phone

Use Expo Go for SDK 57 and scan the QR code. NutriSight normally discovers the same PC that is serving the Expo bundle and uses `http://<that-PC-IP>:8000` automatically.

If the backend is on a different machine, open **Settings -> Backend connection** and enter its full address.

## Important networking note

On a physical Android phone, `localhost` and `127.0.0.1` point to the phone itself, not your PC. That was the cause of the previous `java.net.ConnectException` error.

Both devices must be on the same Wi-Fi for a local backend, and the PC firewall must allow inbound access to Python/port 8000 on the private network.

## Validation performed

- Backend Python syntax: passed
- FastAPI account/profile/history/auth tests: 3/3 passed
- New `/`, `/api`, `/api/ping`, `/health` routes: covered by tests
- Frontend TypeScript/TSX parse/transpile validation: passed
- Expo SDK 57 dependency lock updated with `expo-constants` for LAN host discovery
