"""
family_recognition.py
===========================================================================
Family Member Recognition - a modular OPTIONAL extension to the AI Blind
Navigation Assistant.

This module adds an identity layer on top of the EXISTING YOLO "person"
detections. It does NOT retrain YOLO and does NOT touch the object
detection, distance, priority or navigation logic.

Pipeline (invoked only for "person" detections, at a reduced cadence):

    Person bbox (from YOLO)
        -> InsightFace face detector (SCRFD)
        -> Face embedding (ArcFace, w600k_r50)
        -> Cosine similarity vs registered family embeddings
        -> Best match, thresholded by FACE_RECOGNITION_THRESHOLD
        -> "father" (confident) | "unknown person"   (information only)

Storage (local, under family_data/):
    family_data/members.json        metadata for every member
    family_data/photos/<id>/...     the uploaded photos (kept locally)
    family_data/embeddings/<id>.npy face embeddings (N x 512)

The feature is fully optional. If the face model fails to load, or the
feature is disabled, the rest of the application is unaffected.
===========================================================================
"""

import json
import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

import numpy as np

import config

log = logging.getLogger("family")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[FamilyRecognition] %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)

# Optional native, safe face-detection fallback used only at registration time
# (single-person photo validation) when the InsightFace detector is unavailable.
try:
    import cv2
    _CV_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CV_AVAILABLE = False


class FamilyDBError(Exception):
    """Base error for family-data problems surfaced to the web UI."""


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

class FamilyStore:
    """Persists members, photos and embeddings under family_data/."""

    def __init__(self, base_dir: str = config.FAMILY_DATA_DIR,
                 base_path: Path | None = None) -> None:
        self.base = (base_path or Path(__file__).resolve().parent / base_dir)
        self._lock = threading.Lock()
        self.members_path = self.base / "members.json"
        self._members: dict[str, dict] = {}
        self._version = 0
        self._load()

    # --- metadata ---------------------------------------------------------
    def _load(self) -> None:
        self.base.mkdir(parents=True, exist_ok=True)
        (self.base / "photos").mkdir(exist_ok=True)
        (self.base / "embeddings").mkdir(exist_ok=True)
        if self.members_path.exists():
            try:
                self._members = json.loads(self.members_path.read_text(encoding="utf-8"))
            except Exception as exc:
                log.error("Could not read members.json: %s", exc)
                self._members = {}

    def _save(self) -> None:
        tmp = self.members_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self._members, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        tmp.replace(self.members_path)

    def list_members(self) -> list[dict]:
        with self._lock:
            out = []
            for m in self._members.values():
                emb = self._embedding_count(m["id"])
                out.append({**m, "photo_count": emb,
                            "photos": self._photo_files(m["id"])})
            out.sort(key=lambda m: m.get("created_at", ""))
            return out

    def count(self) -> int:
        """Cheap in-memory member count (no filesystem access).

        ``list_members`` stats files on disk; the real-time pipeline calls
        ``FaceRecognizer.is_available`` on every frame, so it must use this
        instead - otherwise it scans the (OneDrive-synced) family folder at
        the camera frame rate and the video stutters.
        """
        with self._lock:
            return len(self._members)

    def version(self) -> int:
        """Monotonic counter bumped whenever members or embeddings change."""
        return self._version

    def get(self, member_id: str) -> dict | None:
        with self._lock:
            m = self._members.get(member_id)
            if not m:
                return None
            m = {**m, "photo_count": self._embedding_count(member_id)}
            m["photos"] = self._photo_files(member_id)
            return m

    def add(self, name: str, relationship: str,
            embedding: np.ndarray) -> dict:
        member_id = uuid.uuid4().hex[:12]
        member = {
            "id": member_id,
            "name": name.strip(),
            "relationship": relationship.strip(),
            "created_at": time.time(),
        }
        with self._lock:
            self._members[member_id] = member
            self._save()
            self._version += 1
        self._save_embedding(member_id, embedding)
        log.info("Registered family member: %s (%s) - id=%s",
                 member["name"], member["relationship"], member_id)
        return {**member, "photo_count": 1, "photos": self._photo_files(member_id)}

    def add_photo(self, member_id: str, embedding: np.ndarray) -> dict:
        with self._lock:
            if member_id not in self._members:
                raise FamilyDBError("Family member not found.")
        count = self._embedding_count(member_id)
        if count >= config.FAMILY_MAX_PHOTOS:
            raise FamilyDBError(
                f"Maximum of {config.FAMILY_MAX_PHOTOS} photos reached for this member."
            )
        self._save_embedding(member_id, embedding)
        self._version += 1
        log.info("Added photo for %s (total %d)",
                 self._members[member_id]["name"], self._embedding_count(member_id))
        return self.get(member_id)

    def update(self, member_id: str, name: str | None, relationship: str | None) -> dict:
        with self._lock:
            m = self._members.get(member_id)
            if not m:
                raise FamilyDBError("Family member not found.")
            if name is not None and name.strip():
                m["name"] = name.strip()
            if relationship is not None and relationship.strip():
                m["relationship"] = relationship.strip()
            self._save()
        log.info("Updated family member %s", member_id)
        return self.get(member_id)

    def delete(self, member_id: str) -> None:
        self._delete_photo_files(member_id)
        with self._lock:
            if member_id in self._members:
                del self._members[member_id]
                self._save()
                self._version += 1
        log.info("Deleted family member %s", member_id)

    # --- embeddings / photos ----------------------------------------------
    def _embedding_path(self, member_id: str, idx: int = 0) -> Path:
        return self.base / "embeddings" / f"{member_id}_{idx}.npy"

    def _save_embedding(self, member_id: str, embedding: np.ndarray) -> None:
        idx = self._embedding_count(member_id)
        np.save(self._embedding_path(member_id, idx), embedding.astype(np.float32))

    def _embedding_count(self, member_id: str) -> int:
        prefix = f"{member_id}_"
        try:
            return len(list((self.base / "embeddings").glob(f"{prefix}*.npy")))
        except OSError:
            return 0

    def load_embeddings(self) -> dict[str, list[np.ndarray]]:
        """Return {member_id: [embedding, ...]} for all members."""
        result: dict[str, list[np.ndarray]] = {}
        emb_dir = self.base / "embeddings"
        if not emb_dir.exists():
            return result
        for path in emb_dir.glob("*.npy"):
            member_id = path.stem.split("_")[0]
            try:
                arr = np.load(path)
                if arr.ndim == 1 and arr.size:
                    result.setdefault(member_id, []).append(arr.astype(np.float32))
            except Exception:
                log.warning("Skipping unreadable embedding %s", path.name)
        return result

    def _photo_files(self, member_id: str) -> list[str]:
        d = self.base / "photos" / member_id
        if not d.exists():
            return []
        return sorted([p.name for p in d.iterdir() if p.suffix.lower() in
                       {".jpg", ".jpeg", ".png", ".webp"}])

    def _delete_photo_files(self, member_id: str) -> None:
        for p in (self.base / "photos" / member_id).glob("*"):
            try:
                if p.is_file():
                    p.unlink()
            except OSError:
                pass
        for p in (self.base / "embeddings").glob(f"{member_id}_*.npy"):
            try:
                p.unlink()
            except OSError:
                pass

    def save_photo_bytes(self, member_id: str, img_bgr: np.ndarray,
                         ext: str = ".jpg") -> str:
        """Store the uploaded image so the web UI can show it. Returns URL path."""
        d = self.base / "photos" / member_id
        d.mkdir(parents=True, exist_ok=True)
        if not _CV_AVAILABLE:
            return ""
        rel = f"/family-photos/{member_id}/{len(self._photo_files(member_id)) + 1}{ext}"
        full = self.base / "photos" / member_id / (
            f"{len(self._photo_files(member_id)) + 1}{ext}")
        ok, buf = cv2.imencode(ext.replace(".", "."), img_bgr)
        if ok:
            full.write_bytes(buf.tobytes())
            return rel
        return ""


# ---------------------------------------------------------------------------
# Face model (InsightFace FaceAnalysis)
# ---------------------------------------------------------------------------

class FaceModel:
    """Lazy-loaded InsightFace FaceAnalysis instance.

    If it cannot be built (missing model files / dependency failure) the
    rest of the system keeps working: is_ready() returns False and all
    higher-level operations fall back to unknown / non-recognition.
    """

    def __init__(self) -> None:
        self._app = None
        self._error: str | None = None
        self._load_lock = threading.Lock()

    def is_ready(self) -> bool:
        self._ensure()
        return self._app is not None

    @property
    def error(self) -> str | None:
        return self._error

    def _ensure(self) -> None:
        if self._app is not None or self._error:
            return
        with self._load_lock:
            if self._app is not None or self._error:
                return
            try:
                from insightface.app import FaceAnalysis
                root = config.INSIGHTFACE_MODELS_ROOT
                log.info("Loading face model from %s ...", root)
                app = FaceAnalysis(
                    name="buffalo_l", root=str(root),
                    providers=["CPUExecutionProvider"],
                )
                app.prepare(ctx_id=0, det_size=(320, 320))
                self._app = app
                log.info("Face model loaded successfully.")
            except Exception as exc:
                self._error = str(exc)
                log.error("Face model unavailable (%s). Family recognition disabled.", exc)

    def detect_faces(self, img_bgr: np.ndarray) -> list[dict]:
        """Return [{bbox, embedding, face_w, face_h, face_score}]."""
        self._ensure()
        if self._app is None or img_bgr is None or img_bgr.size == 0:
            return []
        try:
            faces = self._app.get(img_bgr)
        except Exception as exc:
            log.warning("Face detection error (skipped): %s", exc)
            return []
        out = []
        for f in faces:
            box = [int(v) for v in f.bbox]
            w = box[2] - box[0]
            h = box[3] - box[1]
            emb = f.normed_embedding
            if emb is None:
                continue
            out.append({
                "bbox": box,
                "embedding": np.asarray(emb, dtype=np.float32),
                "face_w": w,
                "face_h": h,
            })
        return out

    # nfn
    def close(self) -> None:  # pragma: no cover - no-op for interface parity
        self._app = None


# ---------------------------------------------------------------------------
# FaceRecognizer - ties storage + model together with matching & stability
# ---------------------------------------------------------------------------

class FaceRecognizer:
    """High-level API used by the server pipeline and the web UI.

    Responsibilities:
      * register members (validate a single-face photo -> store embedding)
      * recognize the people in a frame (person bbox -> identity/similarity)
      * temporal stability so transients are not announced
      * safe fallbacks when the model is missing
    """

    NAME_BY_ID: dict[str, str] = {}

    def __init__(self) -> None:
        self.store = FamilyStore()
        self.model = FaceModel()
        self.enabled = config.FAMILY_RECOGNITION_ENABLED
        self.threshold = config.FACE_RECOGNITION_THRESHOLD
        self.announce_mode = config.FAMILY_ANNOUNCE_MODE
        self.interval = config.FACE_RECOGNITION_INTERVAL
        self.min_size = config.FACE_MIN_SIZE
        self.stability = config.FACE_RECOGNITION_STABILITY
        self._tick = 0
        self._stability_lock = threading.Lock()
        self._stable: dict[str, str] = {}      # track_id -> current identity
        self._stable_count: dict[str, int] = {}
        self._announced: dict[str, tuple[str, float, str]] = {}  # cooldown per identity
        self._announce_cooldown = config.FAMILY_ANNOUNCE_COOLDOWN_SECONDS
        self._known_cache: dict[str, list[np.ndarray]] | None = None
        self._known_cache_version = -1
        self._load_names()

    # ------------------------------------------------------------------ API
    def is_available(self) -> bool:
        # Cheap in-memory check: this runs on every frame of the pipeline.
        return (self.enabled and self.model.is_ready()
                and self.store.count() > 0)

    def recognize_people(self, img_bgr: np.ndarray,
                         person_boxes: list[dict]) -> list[dict]:
        """Given the raw frame and list of person detections
        [{bbox, confidence}], return one result per person:
        {identity, similarity, member_id, face_w, face_h}.
        Runs at a reduced cadence (self.interval)."""
        if not self.enabled:
            return self._empty_results(person_boxes)
        if not self.model.is_ready():
            return self._empty_results(person_boxes)

        self._tick += 1
        if self._tick % max(self.interval, 1) != 0:
            return self._empty_results(person_boxes)

        known = self._load_known()
        results = []
        for p in person_boxes:
            x1, y1, x2, y2 = p["bbox"]
            face_w = x2 - x1
            if face_w < self.min_size:
                results.append({
                    "identity": None, "similarity": 0.0, "member_id": None,
                    "face_w": 0, "face_h": 0, "reason": "small",
                })
                continue
            # Crop the frame tightly to the upper part of the person box where
            # the face usually is, then run face detection on it.
            crop = self._crop_face_region(img_bgr, x1, y1, x2, y2)
            faces = self.model.detect_faces(crop)
            if not faces:
                results.append({
                    "identity": None, "similarity": 0.0, "member_id": None,
                    "face_w": 0, "face_h": 0, "reason": "no_face",
                })
                continue
            # Use the largest high-quality face in the crop.
            face = max(faces, key=lambda f: f["face_w"] * f["face_h"])
            identity, sim = self._match(face["embedding"], known)
            results.append({
                "identity": identity,
                "similarity": sim,
                "member_id": identity,
                "face_w": face["face_w"],
                "face_h": face["face_h"],
                "reason": "matched" if identity else "low_score",
            })
        return results

    # --- identity helper --------------------------------------------------
    def announce_phrase(self, member_id: str) -> str:
        """Turn a recognized member id into the possessive phrase used in
        the guidance text, following FAMILY_ANNOUNCE_MODE. Returns e.g.
        'your father'."""
        m = self.store.get(member_id) if member_id else None
        if not m:
            return "person"
        rel = m.get("relationship", "")
        name = m.get("name", "")
        rel_lower = rel.lower()
        if self.announce_mode == "name":
            return name or "person"
        if self.announce_mode == "name+relation":
            return f"your {rel_lower} {name}" if rel else name or "person"
        # default: relationship
        return f"your {rel_lower}" if rel else name or "person"

    # --- temporal stability ----------------------------------------------
    def smooth_track(self, track_id: int, identity: str | None) -> str:
        """Return the stable identity for a track, requiring the same
        identity FACE_RECOGNITION_STABILITY times in a row."""
        if not identity:
            with self._stability_lock:
                self._stable_count[track_id] = max(self._stable_count.get(track_id, 0) - 1, 0)
                if self._stable_count.get(track_id, 0) == 0:
                    self._stable.pop(track_id, None)
            return "unknown"

        with self._stability_lock:
            if self._stable.get(track_id) == identity:
                self._stable_count[track_id] = self._stable_count.get(track_id, 0) + 1
            else:
                self._stable[track_id] = identity
                self._stable_count[track_id] = 1
            if self._stable_count[track_id] >= self.stability:
                return identity
            return "unknown"

    def stable_identity(self, track_id: int) -> str:
        """Return the identity currently considered stable for a track,
        WITHOUT mutating the stability counters.

        Needed because face recognition runs at a reduced cadence: the ticks
        in between carry no fresh result, and must not be fed to
        smooth_track() (that would reset the counter so an identity could
        never become stable). Those skipped ticks reuse whatever is already
        stable for the track via this getter."""
        with self._stability_lock:
            if self._stable_count.get(track_id, 0) >= self.stability:
                return self._stable.get(track_id) or "unknown"
            return "unknown"

    # --- re-announcement cooldown ----------------------------------------
    def should_announce(self, member_id: str | None, distance: float,
                        position: str) -> bool:
        """Only re-announce a recognized person after a meaningful change
        (identity, distance bucket, or direction) or after a cooldown."""
        if member_id is None:
            self._announced.pop("person", None)
            return True
        key = member_id
        dist_bucket = round(distance / 0.5) * 0.5
        prev = self._announced.get(key)
        now = time.time()
        if prev is None:
            self._announced[key] = (member_id, dist_bucket, position, now)
            return True
        if prev[2] != position or abs(prev[1] - dist_bucket) > 0.001:
            self._announced[key] = (member_id, dist_bucket, position, now)
            return True
        if now - prev[3] > self._announce_cooldown:
            self._announced[key] = (member_id, dist_bucket, position, now)
            return True
        return False

    # ------------------------------------------------------------------ mod
    def register(self, member_id: str | None, name: str, relationship: str,
                 embeddings: list[np.ndarray],
                 originals: list[tuple[np.ndarray, str]]) -> dict:
        """Create/update a member given pre-computed embeddings (one per
        valid photo). Used by the web registration flow (see /api/family/
        register in server.py, which validates faces before calling this)."""
        if member_id:
            self.store.update(member_id, name, relationship)
            for emb in embeddings:
                self.store.add_photo(member_id, emb)
            member = self.store.get(member_id)
        else:
            member = self.store.add(name, relationship, embeddings[0])
            for emb in embeddings[1:]:
                self.store.add_photo(member["id"], emb)
        # save the original photos for display (local storage only)
        for img, ext in originals:
            if _CV_AVAILABLE:
                self.store.save_photo_bytes(member["id"], img, ext)
        self._load_names()
        return member

    # ---------------------------------------------------------------- utils
    def _empty_results(self, person_boxes: list[dict]) -> list[dict]:
        return [{
            "identity": None, "similarity": 0.0, "member_id": None,
            "face_w": 0, "face_h": 0, "reason": "disabled",
        } for _ in person_boxes]

    def _load_names(self) -> None:
        try:
            for m in self.store.list_members():
                FaceRecognizer.NAME_BY_ID[m["id"]] = m["name"]
        except Exception:
            pass

    def _load_known(self) -> dict[str, list[np.ndarray]]:
        """Registered embeddings, cached until members/embeddings change.

        Reading the .npy files on every recognition tick is disk I/O inside
        the real-time loop; caching keeps the loop smooth while still
        refreshing immediately after a register/delete.
        """
        version = self.store.version()
        if self._known_cache is None or self._known_cache_version != version:
            self._known_cache = self.store.load_embeddings()
            self._known_cache_version = version
        return self._known_cache

    @staticmethod
    def _crop_face_region(img, x1, y1, x2, y2):
        """Expand/prioritise the top of the person box where the face is."""
        h, w = img.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        # take the full person height but pad the top slightly so the head
        # is never cut off.
        top = max(0, y1 - int((y2 - y1) * 0.1))
        bottom = min(h, y2)
        left = max(0, x1)
        right = min(w, x2)
        return img[top:bottom, left:right]

    @staticmethod
    def _match(embedding, known) -> tuple[str | None, float]:
        best_id, best_sim = None, -1.0
        for member_id, embs in known.items():
            for e in embs:
                sim = float(np.dot(embedding, e))
                if sim > best_sim:
                    best_sim, best_id = sim, member_id
        if best_id is not None and best_sim >= config.FACE_RECOGNITION_THRESHOLD:
            return best_id, best_sim
        return None, best_sim if best_id is not None else 0.0


# ---------------------------------------------------------------------------
# Registration-time face validation (single-face, strict)
# ---------------------------------------------------------------------------

def validate_uploaded_image(img_bgr: np.ndarray, model: FaceModel) -> dict:
    """Validate an uploaded registration photo:
        - decodes to a valid image
        - contains EXACTLY ONE face, large enough to be useful
    Returns {ok, faces, error} where faces carries embeddings for OK photos.

    IMPORTANT: never registers a photo with 0 faces (no identity to
    recognise) or >1 face (ambiguous identity).
    """
    if img_bgr is None or img_bgr.size == 0:
        return {"ok": False, "faces": [], "error": "Invalid or empty image."}
    if not model.is_ready():
        return {"ok": False, "faces": [], "error":
                "Face model is not available; cannot validate the photo."}

    faces = model.detect_faces(img_bgr)
    if len(faces) == 0:
        return {"ok": False, "faces": [], "error":
                "No face detected in this photo. Please upload a clear, "
                "front-facing photo of the person."}
    if len(faces) > 1:
        return {"ok": False, "faces": [], "error":
                f"Multiple faces detected ({len(faces)}). Please upload a "
                "photo containing only this family member."}

    return {"ok": True, "faces": faces, "error": None}
