"""NutriSight API: accounts, profile, private history and real model inference."""

import hashlib
import hmac
import io
import json
import logging
import os
import re
import secrets
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from threading import Lock

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from .pipeline import FoodPipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.getenv("NUTRISIGHT_DATA_DIR", str(ROOT / "data")))
DATA.mkdir(parents=True, exist_ok=True)
DB = DATA / "nutrisight.sqlite3"
MAX_UPLOAD = 10 * 1024 * 1024
MAX_PIXELS = 24_000_000
pipeline = FoodPipeline()
logger = logging.getLogger("nutrisight")


@contextmanager
def db():
    conn = sqlite3.connect(DB, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users(
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                password TEXT NOT NULL,
                profile TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS sessions(
                token TEXT PRIMARY KEY,
                user_id TEXT REFERENCES users(id) ON DELETE CASCADE,
                expires REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS scans(
                id TEXT PRIMARY KEY,
                user_id TEXT REFERENCES users(id) ON DELETE CASCADE,
                created REAL NOT NULL,
                result TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS scans_user ON scans(user_id, created);
            """
        )


@asynccontextmanager
async def lifespan(app):
    init_db()
    yield


app = FastAPI(title="NutriSight API", version="3.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "CORS_ORIGINS",
        "http://localhost:8081,http://localhost:19006,http://localhost:8080",
    ).split(","),
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})(?::\d+)?",
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-File-Name"],
)


def password_hash(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.scrypt(
        password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1
    ).hex()
    return salt + ":" + digest


def current_user(authorization: str | None = Header(default=None)):
    if not authorization:
        return None
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Please sign in again.")
    digest = hashlib.sha256(authorization[7:].encode()).hexdigest()
    with db() as conn:
        row = conn.execute(
            "SELECT u.* FROM users u JOIN sessions s ON u.id=s.user_id "
            "WHERE s.token=? AND s.expires>?",
            (digest, time.time()),
        ).fetchone()
    if not row:
        raise HTTPException(401, "Your session expired. Please sign in again.")
    return dict(row)


def required_user(user=Depends(current_user)):
    if not user:
        raise HTTPException(401, "Please sign in first.")
    return user


def public_user(user):
    profile = json.loads(user["profile"] or "{}")
    return {
        "id": user["id"],
        "email": user["email"],
        "name": user["name"],
        "profile": profile,
    }


def create_session(user):
    token = secrets.token_urlsafe(48)
    with db() as conn:
        conn.execute("DELETE FROM sessions WHERE expires<=?", (time.time(),))
        conn.execute(
            "INSERT INTO sessions VALUES(?,?,?)",
            (
                hashlib.sha256(token.encode()).hexdigest(),
                user["id"],
                time.time() + 30 * 86400,
            ),
        )
    return {"token": token, "user": public_user(user)}


class Credentials(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(min_length=8, max_length=128)
    name: str = Field(default="", max_length=80)


class Profile(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    age: int | None = Field(default=None, ge=1, le=120)
    sex: str | None = Field(default=None, max_length=30)
    height_cm: float | None = Field(default=None, ge=50, le=260)
    weight_kg: float | None = Field(default=None, ge=15, le=400)
    allergies: list[str] = Field(default_factory=list, max_length=30)
    conditions: list[str] = Field(default_factory=list, max_length=30)
    preferences: list[str] = Field(default_factory=list, max_length=30)
    avoid: list[str] = Field(default_factory=list, max_length=30)
    goal: str = Field(default="Build mindful habits", max_length=200)

    def clean(self):
        data = self.model_dump()
        data["name"] = data["name"].strip()
        if not data["name"]:
            raise HTTPException(422, "Please enter your name.")
        if data.get("sex"):
            data["sex"] = data["sex"].strip()[:30]
        for key in ["allergies", "conditions", "preferences", "avoid"]:
            data[key] = list(
                dict.fromkeys(item.strip()[:80] for item in data[key] if item.strip())
            )
        data["goal"] = data["goal"].strip() or "Build mindful habits"
        data["onboarding_complete"] = True
        data["onboarding_skipped"] = False
        return data


class PasswordChange(BaseModel):
    current_password: str = Field(min_length=8, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


def email_normalize(value):
    value = value.strip().lower()
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value):
        raise HTTPException(422, "Enter a valid email address.")
    return value


_attempts = {}
_attempt_lock = Lock()


def login_limit(email):
    now = time.time()
    with _attempt_lock:
        for key in list(_attempts):
            _attempts[key] = [t for t in _attempts[key] if now - t < 60]
            if not _attempts[key]:
                del _attempts[key]
        entries = _attempts.setdefault(email, [])
        if len(entries) >= 10:
            raise HTTPException(429, "Too many attempts. Try again in a minute.")
        entries.append(now)


@app.post("/api/auth/signup", status_code=201)
def signup(body: Credentials):
    email = email_normalize(body.email)
    login_limit(email)
    name = body.name.strip()
    if not name:
        raise HTTPException(422, "Please enter your name.")
    user = {
        "id": uuid.uuid4().hex,
        "email": email,
        "name": name,
        "password": password_hash(body.password),
        "profile": "{}",
    }
    try:
        with db() as conn:
            conn.execute(
                "INSERT INTO users VALUES(:id,:email,:name,:password,:profile)", user
            )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(
            409, "This email already has an account. Please sign in."
        ) from exc
    return create_session(user)


@app.post("/api/auth/login")
def login(body: Credentials):
    email = email_normalize(body.email)
    login_limit(email)
    with db() as conn:
        row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    salt = row["password"].split(":")[0] if row else "00" * 16
    candidate = password_hash(body.password, salt)
    if not row or not hmac.compare_digest(candidate, row["password"]):
        raise HTTPException(401, "Email or password is incorrect.")
    return create_session(dict(row))


@app.post("/api/auth/logout")
def logout(authorization: str = Header(), user=Depends(required_user)):
    with db() as conn:
        conn.execute(
            "DELETE FROM sessions WHERE token=?",
            (hashlib.sha256(authorization[7:].encode()).hexdigest(),),
        )
    return {"ok": True}


@app.get("/api/me")
def me(user=Depends(required_user)):
    return public_user(user)


@app.put("/api/profile")
def update_profile(body: Profile, user=Depends(required_user)):
    data = body.clean()
    with db() as conn:
        conn.execute(
            "UPDATE users SET name=?, profile=? WHERE id=?",
            (data["name"], json.dumps(data), user["id"]),
        )
    user.update(name=data["name"], profile=json.dumps(data))
    return public_user(user)


@app.post("/api/profile/skip")
def skip_profile(user=Depends(required_user)):
    existing = json.loads(user["profile"] or "{}")
    existing.update(
        {
            "name": user["name"],
            "age": existing.get("age"),
            "sex": existing.get("sex"),
            "height_cm": existing.get("height_cm"),
            "weight_kg": existing.get("weight_kg"),
            "allergies": existing.get("allergies", []),
            "conditions": existing.get("conditions", []),
            "preferences": existing.get("preferences", []),
            "avoid": existing.get("avoid", []),
            "goal": existing.get("goal", "Build mindful habits"),
            "onboarding_complete": True,
            "onboarding_skipped": True,
        }
    )
    with db() as conn:
        conn.execute(
            "UPDATE users SET profile=? WHERE id=?",
            (json.dumps(existing), user["id"]),
        )
    user["profile"] = json.dumps(existing)
    return public_user(user)


@app.delete("/api/me")
def delete_account(user=Depends(required_user)):
    with db() as conn:
        conn.execute("DELETE FROM users WHERE id=?", (user["id"],))
    return {"ok": True}


@app.put("/api/auth/password")
def change_password(body: PasswordChange, user=Depends(required_user)):
    current = password_hash(body.current_password, user["password"].split(":")[0])
    if not hmac.compare_digest(current, user["password"]):
        raise HTTPException(400, "Current password is incorrect.")
    user["password"] = password_hash(body.new_password)
    with db() as conn:
        conn.execute(
            "UPDATE users SET password=? WHERE id=?", (user["password"], user["id"])
        )
        conn.execute("DELETE FROM sessions WHERE user_id=?", (user["id"],))
    return create_session(user)


@app.get("/health")
def health():
    return {"status": "ok", "api_version": "3.0.0", "models": pipeline.status()}


# Conservative name matching. This never claims an image proves a meal is allergen-free.
def dietary_insights(foods, profile):
    if not profile:
        return []
    if not foods:
        return [
            {
                "level": "info",
                "title": "No foods to check",
                "reason": "Try a clearer photo before reviewing dietary insights.",
            }
        ]
    names = " ".join(
        f["name"].lower().replace("_", " ").replace("-", " ") for f in foods
    )
    aliases = {
        "milk": ["milk", "cheese", "yogurt", "yoghurt", "cream", "butter"],
        "dairy": ["milk", "cheese", "yogurt", "yoghurt", "cream", "butter"],
        "peanut": ["peanut", "groundnut"],
        "egg": ["egg", "omelet", "omelette", "omlete", "ডিম"],
        "wheat": ["wheat", "bread", "pasta", "noodle"],
        "shellfish": ["shrimp", "prawn", "crab", "lobster"],
        "soy": ["soy", "tofu"],
        "fish": ["fish", "salmon", "tuna", "ইলিশ", "মাছ"],
    }
    alerts = []
    if any(food["name"] == "unknown" for food in foods):
        alerts.append(
            {
                "level": "warning",
                "title": "Some foods are uncertain",
                "reason": "Unknown foods cannot be checked against your preferences. Verify ingredients.",
            }
        )
    for saved in profile.get("allergies", []) + profile.get("avoid", []):
        words = aliases.get(saved.lower(), [saved.lower()])
        if any(re.search(r"\b" + re.escape(word) + r"s?\b", names) for word in words):
            alerts.append(
                {
                    "level": "warning",
                    "title": f"Check for {saved}",
                    "reason": f"You marked {saved} in your profile. A detected food name may match. Verify ingredients before eating.",
                }
            )
    preferences = [item.lower() for item in profile.get("preferences", [])]
    if any(item in preferences for item in ["vegetarian", "vegan"]):
        terms = ["chicken", "beef", "pork", "fish", "lamb", "mutton", "shrimp", "meat"]
        if "vegan" in preferences:
            terms += ["milk", "cheese", "egg", "butter", "yogurt"]
        if any(re.search(r"\b" + re.escape(word) + r"\b", names) for word in terms):
            alerts.append(
                {
                    "level": "warning",
                    "title": "Review your dietary preference",
                    "reason": "A detected food may conflict with your saved vegetarian or vegan preference.",
                }
            )
    preference_checks = {
        "no pork": ["pork"],
        "no beef": ["beef"],
        "halal": ["pork"],
    }
    for preference, terms in preference_checks.items():
        if preference in preferences and any(re.search(r"\b" + re.escape(word) + r"\b", names) for word in terms):
            alerts.append(
                {
                    "level": "warning",
                    "title": f"Review your {preference} preference",
                    "reason": f"A detected food name may conflict with your saved {preference} preference. Verify the ingredients and preparation.",
                }
            )
    if not alerts:
        alerts.append(
            {
                "level": "info",
                "title": "No name-based conflict found",
                "reason": "This is not confirmation that the meal is allergen-free. Hidden ingredients and incorrect predictions remain possible.",
            }
        )
    if profile.get("conditions"):
        alerts.append(
            {
                "level": "info",
                "title": "Conditions saved for your reference",
                "reason": "NutriSight does not diagnose conditions or prescribe medical diets. Follow your clinician’s guidance.",
            }
        )
    return alerts


def _decode_image(raw: bytes):
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "Choose an image smaller than 10 MB.")
    try:
        image = Image.open(io.BytesIO(raw))
        if image.width * image.height > MAX_PIXELS:
            raise HTTPException(413, "Choose an image below 24 megapixels.")
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.load()
        return image
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, "Upload a valid JPG, PNG or WebP image.") from exc


def _run_analysis(raw: bytes, user):
    image = _decode_image(raw)
    try:
        result = pipeline.analyze(image)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(
            503,
            "The analysis service could not run. Check /health and the backend terminal, then try again.",
        ) from exc

    profile = json.loads(user["profile"] or "{}") if user else {}
    onboarding_complete = bool(profile.get("onboarding_complete")) or bool(profile)
    result.update(
        id=uuid.uuid4().hex,
        created_at=time.time(),
        alerts=dietary_insights(result["foods"], profile),
        profile_required=not onboarding_complete,
        saved=bool(user),
    )
    if user:
        with db() as conn:
            conn.execute(
                "INSERT INTO scans VALUES(?,?,?,?)",
                (result["id"], user["id"], result["created_at"], json.dumps(result)),
            )
    return result


@app.post("/api/analyze")
def analyze_multipart(image: UploadFile = File(...), user=Depends(current_user)):
    raw = image.file.read(MAX_UPLOAD + 1)
    return _run_analysis(raw, user)


@app.post("/api/analyze/raw")
async def analyze_raw(request: Request, user=Depends(current_user)):
    """Native Expo-friendly upload endpoint that avoids multipart/FormData quirks."""
    content_type = (request.headers.get("content-type") or "").split(";", 1)[0].lower()
    if content_type not in {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "application/octet-stream",
    }:
        raise HTTPException(415, "Send a JPG, PNG or WebP image.")
    raw = await request.body()
    return _run_analysis(raw, user)


@app.get("/api/history")
def history(limit: int = 50, offset: int = 0, user=Depends(required_user)):
    with db() as conn:
        rows = conn.execute(
            "SELECT result FROM scans WHERE user_id=? ORDER BY created DESC LIMIT ? OFFSET ?",
            (user["id"], min(max(limit, 1), 100), max(offset, 0)),
        ).fetchall()
    return [json.loads(row["result"]) for row in rows]


@app.delete("/api/history/{scan_id}")
def delete_scan(scan_id: str, user=Depends(required_user)):
    with db() as conn:
        cur = conn.execute(
            "DELETE FROM scans WHERE id=? AND user_id=?", (scan_id, user["id"])
        )
        if not cur.rowcount:
            raise HTTPException(404, "Scan not found.")
    return {"ok": True}


@app.get("/api/dashboard")
def dashboard(user=Depends(required_user)):
    with db() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM scans WHERE user_id=?", (user["id"],)
        ).fetchone()[0]
        recent = conn.execute(
            "SELECT result FROM scans WHERE user_id=? ORDER BY created DESC LIMIT 3",
            (user["id"],),
        ).fetchall()
    return {"total_scans": total, "recent": [json.loads(row["result"]) for row in recent]}
