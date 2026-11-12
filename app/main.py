"""
AutoFix AI - FastAPI backend (Python 3.11, CPU-only).

Endpoints
---------
POST /scan               OBD-II telemetry -> XGBoost fault class -> Gemini repair steps
                         (non-normal states only) -> async BigQuery logging.
POST /post-repair-check  Compare pre/post-repair telemetry and verify normalization.
GET  /health             Liveness/readiness probe (always HTTP 200 once started).

Design notes
------------
* Non-blocking: XGBoost inference and the (synchronous) BigQuery client run in
  small dedicated thread pools; the Gemini call uses the SDK's native async client.
  The event loop never does CPU or network-blocking work.
* Low CPU: model is loaded once at startup, pinned to 1 inference thread
  (n_jobs=1) and requests are served from a small bounded pool. No GPU libs.
* LLM safety net: hard timeout, concurrency cap, circuit breaker and a static
  fallback guide, so /scan never fails or hangs because Gemini is slow/down.
* DPDP Act 2023 / zero-PII posture:
    - The API accepts no names, phones, emails, locations or free text on /scan.
    - The optional vehicle_id is HMAC-SHA256 pseudonymised before persistence;
      the raw value is never logged, stored, sent to the LLM or echoed back.
    - Only numeric telemetry + fault label go to Gemini.
    - /post-repair-check free-text repair actions are used for validation only;
      they are never logged, stored or echoed.
    - Every log record passes through a PII masker (email, phone, Aadhaar, PAN,
      VIN, Indian plate, IPv4) as defence in depth, and validation errors are
      returned WITHOUT echoing the submitted input.
    - Data residency: choose asia-south1/asia-south2 for BigQuery (and Vertex AI,
      where the model is available) if you need in-India processing.

Environment variables (all optional unless noted)
-------------------------------------------------
APP_ENV                 development | production   (production hides /docs and
                        REQUIRES PII_HASH_SALT)
PII_HASH_SALT           Secret salt for vehicle_id pseudonymisation (use Secret Manager)
MODEL_PATH              Path to model.joblib (default: <repo>/model.joblib)
GCP_PROJECT_ID          Project for Vertex AI + BigQuery
GCP_LOCATION            Vertex AI region (default us-central1)
GEMINI_API_KEY          Use the Gemini Developer API instead of Vertex AI
GEMINI_MODEL            Default gemini-3.8-flash (Gemini 1.5 Flash is retired by Google)
LLM_TIMEOUT_S           Gemini deadline in seconds (default 8)
BQ_LOGGING_ENABLED      true|false (default true; set false on your laptop)
BQ_DATASET / BQ_TABLE   Default autofix_db / telemetry_logs
MIN_CONFIDENCE          Below this, responses carry low_confidence=true (default 0.55)

Deployment note (Cloud Run)
---------------------------
BackgroundTasks run AFTER the response is sent. With default request-based CPU
billing, Cloud Run throttles CPU once the response is flushed, which can stall
the BigQuery insert. Deploy with `--no-cpu-throttling` (instance-based billing),
or swap the background task for a Pub/Sub publish if you want strict at-least-once.
Run uvicorn with `--no-access-log` (access logs carry client IPs).
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Callable, Literal

import joblib
import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from google import genai
from google.cloud import bigquery
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

APP_VERSION = "1.0.0"
NORMAL_LABEL = "Normal"
MAX_BODY_BYTES = 16 * 1024  # telemetry payloads are tiny; reject anything larger

# =============================================================================
# Logging: structured JSON for Cloud Logging + global PII masking
# =============================================================================
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "[EMAIL]"),
    (re.compile(r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)"), "[AADHAAR]"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.IGNORECASE), "[PAN]"),
    (re.compile(r"\b[A-Z]{2}[ -]?\d{1,2}[ -]?[A-Z]{1,3}[ -]?\d{4}\b", re.IGNORECASE), "[PLATE]"),
    (re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b", re.IGNORECASE), "[VIN]"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[IP]"),
    (re.compile(r"(?<![\w.])(?:\+91[ -]?|0)?[6-9]\d{9}(?![\w.])"), "[PHONE]"),
    (re.compile(r"(?<![\w.])\+\d{1,3}[ -]?\d{6,12}(?![\w.])"), "[PHONE]"),
]


def mask_pii(text: str) -> str:
    """Replace anything that looks like personal data with a placeholder."""
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class JsonFormatter(logging.Formatter):
    """One JSON object per line; `severity`/`message` are understood by Cloud Logging."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "time": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "request_id": request_id_var.get(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            for key, value in fields.items():
                payload[key] = mask_pii(value) if isinstance(value, str) else value
        if record.exc_info:
            payload["exception"] = mask_pii(self.formatException(record.exc_info))
        return json.dumps(payload, default=str, ensure_ascii=False)


_logging_configured = False


def configure_logging() -> None:
    """Install JSON logging and a record factory that masks PII in every message."""
    global _logging_configured
    if _logging_configured:
        return
    _logging_configured = True

    previous_factory = logging.getLogRecordFactory()

    def masking_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = previous_factory(*args, **kwargs)
        try:
            message = record.getMessage()
        except Exception:  # malformed format args must never break logging
            message = str(record.msg)
        record.msg = mask_pii(message)
        record.args = None
        return record

    logging.setLogRecordFactory(masking_factory)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

    # Route uvicorn's loggers through the same JSON/masking pipeline.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers = []
        uv_logger.propagate = True


configure_logging()
log = logging.getLogger("autofix")


def log_event(level: int, message: str, **fields: Any) -> None:
    """Structured log helper. Never pass raw user input as a field."""
    log.log(level, message, extra={"fields": fields})


# =============================================================================
# Settings
# =============================================================================
def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_env: str
    model_path: Path
    gcp_project: str | None
    gcp_location: str
    gemini_model: str
    gemini_api_key: str | None
    llm_timeout_s: float
    llm_max_concurrency: int
    breaker_threshold: int
    breaker_cooldown_s: float
    bq_enabled: bool
    bq_dataset: str
    bq_table: str
    bq_timeout_s: float
    bq_max_attempts: int
    bq_max_concurrency: int
    inference_workers: int
    min_confidence: float
    pii_salt: bytes


def load_settings() -> Settings:
    app_env = os.getenv("APP_ENV", "development").strip().lower()

    salt_raw = os.getenv("PII_HASH_SALT")
    if salt_raw:
        pii_salt = salt_raw.encode("utf-8")
    elif app_env == "production":
        raise RuntimeError("PII_HASH_SALT must be set when APP_ENV=production")
    else:
        # Dev only: per-process salt. Hashes will not be stable across restarts.
        pii_salt = secrets.token_bytes(32)
        log_event(logging.WARNING, "PII_HASH_SALT not set; using ephemeral salt (development only)")

    default_model = Path(__file__).resolve().parent.parent / "model.joblib"
    return Settings(
        app_env=app_env,
        model_path=Path(os.getenv("MODEL_PATH", str(default_model))),
        gcp_project=os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT"),
        gcp_location=os.getenv("GCP_LOCATION", "us-central1"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        llm_timeout_s=float(os.getenv("LLM_TIMEOUT_S", "8")),
        llm_max_concurrency=int(os.getenv("LLM_MAX_CONCURRENCY", "8")),
        breaker_threshold=int(os.getenv("LLM_BREAKER_THRESHOLD", "5")),
        breaker_cooldown_s=float(os.getenv("LLM_BREAKER_COOLDOWN_S", "30")),
        bq_enabled=_env_bool("BQ_LOGGING_ENABLED", True),
        bq_dataset=os.getenv("BQ_DATASET", "autofix_db"),
        bq_table=os.getenv("BQ_TABLE", "telemetry_logs"),
        bq_timeout_s=float(os.getenv("BQ_TIMEOUT_S", "10")),
        bq_max_attempts=int(os.getenv("BQ_MAX_ATTEMPTS", "3")),
        bq_max_concurrency=int(os.getenv("BQ_MAX_CONCURRENCY", "2")),
        inference_workers=int(os.getenv("INFERENCE_WORKERS", "2")),
        min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.55")),
        pii_salt=pii_salt,
    )


# =============================================================================
# Pydantic schemas (strict: unknown fields rejected, NaN/Inf rejected)
# =============================================================================
# Bounds equal the envelope the classifier was trained on (train_model.py).
# Rejecting out-of-range values avoids silent model extrapolation. Note: this
# means a cold-engine scan (< 70 C) is rejected until the model is retrained
# with cold-start data.
class TelemetryReading(BaseModel):
    """SAE J1979 Mode 01 engine telemetry snapshot."""

    model_config = ConfigDict(extra="forbid")

    rpm: float = Field(..., ge=600, le=4000, allow_inf_nan=False,
                       description="PID 0x0C engine speed (rpm)")
    coolant_temp_c: float = Field(..., ge=70, le=115, allow_inf_nan=False,
                                  description="PID 0x05 coolant temperature (deg C)")
    short_fuel_trim_pct: float = Field(..., ge=-25, le=25, allow_inf_nan=False,
                                       description="PID 0x06 short-term fuel trim (%)")
    long_fuel_trim_pct: float = Field(..., ge=-25, le=25, allow_inf_nan=False,
                                      description="PID 0x07 long-term fuel trim (%)")
    intake_pressure_kpa: float = Field(..., ge=20, le=100, allow_inf_nan=False,
                                       description="PID 0x0B intake manifold pressure (kPa)")
    engine_load_pct: float = Field(..., ge=0, le=100, allow_inf_nan=False,
                                   description="PID 0x04 calculated engine load (%)")


class ScanRequest(TelemetryReading):
    # Identifiers are restricted to a safe charset. Send a pseudonymous id, not a raw VIN.
    vehicle_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{6,64}$",
                                   description="Pseudonymous vehicle identifier (hashed before storage)")
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{8,64}$",
                                   description="Client diagnostic session id")


class ScanResponse(BaseModel):
    request_id: str
    fault_state: str
    confidence: float = Field(..., ge=0, le=1)
    probabilities: dict[str, float]
    is_fault: bool
    low_confidence: bool
    repair_steps: list[str]
    safety_notice: str | None = None
    repair_source: Literal["gemini", "fallback", "not_required"]
    model_version: str


RepairAction = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class PostRepairRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pre_repair: TelemetryReading
    repair_actions: list[RepairAction] = Field(..., min_length=1, max_length=20,
                                               description="Actions performed (never logged or stored)")
    post_repair: TelemetryReading
    vehicle_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{6,64}$")
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{8,64}$")


class ResolutionStatus(str, Enum):
    RESOLVED = "RESOLVED"
    PARTIALLY_RESOLVED = "PARTIALLY_RESOLVED"
    UNRESOLVED = "UNRESOLVED"
    REGRESSED = "REGRESSED"
    NO_PRIOR_ANOMALY = "NO_PRIOR_ANOMALY"


class AnomalyDetail(BaseModel):
    parameter: str
    unit: str
    observed_value: float
    healthy_min: float
    healthy_max: float
    deviation: float = Field(..., description="Distance outside the healthy band")
    pre_repair_deviation: float | None = None
    improved: bool
    newly_introduced: bool


class PostRepairClassification(BaseModel):
    fault_state: str
    confidence: float


class PostRepairResponse(BaseModel):
    request_id: str
    normalization_pct: float = Field(..., ge=0, le=100)
    resolution_status: ResolutionStatus
    pre_repair_anomalies: list[str]
    remaining_anomalies: list[AnomalyDetail]
    post_repair_classification: PostRepairClassification
    notes: list[str]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    llm_available: bool
    bigquery_logging: bool
    version: str


class RepairGuide(BaseModel):
    """Structured output schema requested from Gemini (kept constraint-free for the API)."""

    steps: list[str]
    safety_notice: str


# =============================================================================
# Runtime container (model, clients, pools, limiters)
# =============================================================================
class CircuitBreaker:
    """Skips the LLM for a cooldown after N consecutive failures (protects latency)."""

    def __init__(self, threshold: int, cooldown_s: float) -> None:
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.failures = 0
        self.open_until = 0.0

    def allow(self) -> bool:
        return time.monotonic() >= self.open_until

    def record_success(self) -> None:
        self.failures = 0

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.open_until = time.monotonic() + self.cooldown_s
            self.failures = 0
            log_event(logging.WARNING, "LLM circuit breaker opened", cooldown_s=self.cooldown_s)


@dataclass
class LoadedModel:
    model: Any
    features: list[str]
    labels: list[str]
    version: str


@dataclass
class Runtime:
    settings: Settings
    loaded: LoadedModel
    inference_executor: ThreadPoolExecutor
    bq_executor: ThreadPoolExecutor
    genai_client: genai.Client | None
    llm_config: types.GenerateContentConfig | None
    llm_semaphore: asyncio.Semaphore
    breaker: CircuitBreaker
    bq_client: bigquery.Client | None
    bq_table_id: str
    bq_semaphore: asyncio.Semaphore


def _load_bundle(path: Path) -> LoadedModel:
    """Load and validate the joblib bundle written by train_model.py.

    SECURITY: joblib uses pickle. Only load artifacts produced by your own
    pipeline (e.g. from a private bucket baked into the image), never user uploads.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Model bundle not found at {path}")

    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or not {"model", "features", "labels", "version"} <= bundle.keys():
        raise ValueError("model.joblib must be a dict with keys: model, features, labels, version")

    model, features, labels = bundle["model"], list(bundle["features"]), list(bundle["labels"])
    unknown = [f for f in features if f not in TelemetryReading.model_fields]
    if unknown:
        raise ValueError(f"Model features missing from API schema: {unknown}")
    if NORMAL_LABEL not in labels:
        raise ValueError(f"Label '{NORMAL_LABEL}' missing from model labels")
    if not hasattr(model, "predict_proba") or len(getattr(model, "classes_", [])) != len(labels):
        raise ValueError("Model is not a fitted classifier matching the label list")

    # One thread per prediction: concurrency comes from our pool, not XGBoost's.
    try:
        model.set_params(n_jobs=1)
        model.get_booster().set_param({"nthread": 1})
    except Exception:  # non-XGBoost estimator: nothing to pin
        pass

    # Warm-up: initialises XGBoost internals so the first request is fast.
    warm = pd.DataFrame([[0.0] * len(features)], columns=features, dtype="float32")
    model.predict_proba(warm)

    return LoadedModel(model=model, features=features, labels=labels, version=str(bundle["version"]))


def _init_bigquery(settings: Settings) -> bigquery.Client | None:
    if not settings.bq_enabled:
        log_event(logging.INFO, "BigQuery logging disabled by configuration")
        return None
    if not settings.gcp_project:
        log_event(logging.WARNING, "GCP_PROJECT_ID not set; BigQuery logging disabled")
        return None
    try:
        return bigquery.Client(project=settings.gcp_project)
    except Exception as exc:  # e.g. no Application Default Credentials on a laptop
        log_event(logging.WARNING, "BigQuery client unavailable; logging disabled",
                  error_type=type(exc).__name__)
        return None


SYSTEM_INSTRUCTION = (
    "You are a senior automotive diagnostic technician assisting a workshop. "
    "You receive OBD-II readings and a machine-detected fault class. "
    "Respond ONLY with JSON matching the schema. Provide 4 to 8 concise, ordered, "
    "actionable repair/inspection steps (each under 240 characters), from cheapest/"
    "safest checks to more invasive ones, and finish with a verification rescan step. "
    "Use only the data provided; do not invent part numbers, prices or readings. "
    "Put any critical safety warning in safety_notice (one or two sentences)."
)


def _init_genai(settings: Settings) -> tuple[genai.Client | None, types.GenerateContentConfig | None]:
    http_options = types.HttpOptions(timeout=int(settings.llm_timeout_s * 1000))  # milliseconds
    try:
        if settings.gemini_api_key:
            client = genai.Client(api_key=settings.gemini_api_key, http_options=http_options)
        elif settings.gcp_project:
            client = genai.Client(vertexai=True, project=settings.gcp_project,
                                  location=settings.gcp_location, http_options=http_options)
        else:
            log_event(logging.WARNING, "No Gemini credentials/project configured; using fallback guides")
            return None, None
    except Exception as exc:
        log_event(logging.WARNING, "Gemini client unavailable; using fallback guides",
                  error_type=type(exc).__name__)
        return None, None

    config_kwargs: dict[str, Any] = {
        "system_instruction": SYSTEM_INSTRUCTION,
        "temperature": 0.2,
        "max_output_tokens": 900,
        "response_mime_type": "application/json",
        "response_schema": RepairGuide,
    }
    # Disable "thinking" on 2.5-series Flash: lower latency and cost for this task.
    if "2.5" in settings.gemini_model and "flash" in settings.gemini_model:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    return client, types.GenerateContentConfig(**config_kwargs)


async def build_runtime(settings: Settings) -> Runtime:
    loop = asyncio.get_running_loop()
    inference_executor = ThreadPoolExecutor(max_workers=settings.inference_workers,
                                            thread_name_prefix="infer")
    bq_executor = ThreadPoolExecutor(max_workers=settings.bq_max_concurrency,
                                     thread_name_prefix="bq")

    loaded = await loop.run_in_executor(inference_executor, _load_bundle, settings.model_path)
    bq_client = await loop.run_in_executor(bq_executor, _init_bigquery, settings)
    genai_client, llm_config = _init_genai(settings)

    log_event(logging.INFO, "Runtime ready", model_version=loaded.version,
              llm_enabled=genai_client is not None, bq_enabled=bq_client is not None,
              gemini_model=settings.gemini_model)

    return Runtime(
        settings=settings,
        loaded=loaded,
        inference_executor=inference_executor,
        bq_executor=bq_executor,
        genai_client=genai_client,
        llm_config=llm_config,
        llm_semaphore=asyncio.Semaphore(settings.llm_max_concurrency),
        breaker=CircuitBreaker(settings.breaker_threshold, settings.breaker_cooldown_s),
        bq_client=bq_client,
        bq_table_id=f"{settings.gcp_project}.{settings.bq_dataset}.{settings.bq_table}",
        bq_semaphore=asyncio.Semaphore(settings.bq_max_concurrency),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    runtime = await build_runtime(settings)  # fail fast: bad model => container never becomes ready
    app.state.runtime = runtime
    try:
        yield
    finally:
        runtime.inference_executor.shutdown(wait=False, cancel_futures=True)
        runtime.bq_executor.shutdown(wait=True)  # let in-flight inserts finish
        if runtime.bq_client is not None:
            runtime.bq_client.close()
        if runtime.genai_client is not None:
            try:
                await runtime.genai_client.aio.aclose()
            except Exception:
                pass


def get_runtime(request: Request) -> Runtime:
    return request.app.state.runtime


RuntimeDep = Annotated[Runtime, Depends(get_runtime)]


# =============================================================================
# Inference
# =============================================================================
@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


def _predict_sync(loaded: LoadedModel, reading: TelemetryReading) -> Prediction:
    # DataFrame with named columns: the model was trained with feature names.
    row = [[float(getattr(reading, name)) for name in loaded.features]]
    frame = pd.DataFrame(row, columns=loaded.features, dtype="float32")
    proba = loaded.model.predict_proba(frame)[0]
    best = int(proba.argmax())
    return Prediction(
        label=loaded.labels[best],
        confidence=round(float(proba[best]), 4),
        probabilities={label: round(float(p), 4) for label, p in zip(loaded.labels, proba)},
    )


async def classify(rt: Runtime, reading: TelemetryReading) -> Prediction:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(rt.inference_executor, _predict_sync, rt.loaded, reading)


# =============================================================================
# Repair-step generation (Gemini Flash with timeout, breaker and static fallback)
# =============================================================================
_FALLBACK_GUIDES: dict[str, tuple[list[str], str]] = {
    "Fuel System Lean": (
        [
            "Read stored DTCs (typically P0171/P0174) and note freeze-frame data.",
            "Check for vacuum or intake leaks, since they mimic a lean fuel condition.",
            "Inspect and clean the MAF sensor; check its reading against expected airflow.",
            "Test fuel pressure and volume at the rail against the OEM specification.",
            "Inspect the fuel filter, pump output and injectors for restriction or clogging.",
            "Check upstream O2 sensor response; replace if slow or biased lean.",
            "Clear codes, drive a full cycle and rescan to confirm fuel trims return within +/-10%.",
        ],
        "Fuel leaks are a fire hazard. Relieve fuel pressure before disconnecting fuel lines.",
    ),
    "Cooling Fault": (
        [
            "Let the engine cool fully, then check coolant level and condition.",
            "Inspect hoses, radiator and water pump area for leaks or blockage.",
            "Verify the radiator fan(s) switch on at the expected temperature.",
            "Test the thermostat: stuck-closed causes overheating, stuck-open prevents warm-up.",
            "Check the coolant temperature sensor against an infrared thermometer reading.",
            "Pressure-test the cooling system and cap; bleed trapped air after refilling.",
            "Rescan at operating temperature to confirm coolant stabilises near 85-100 C.",
        ],
        "Never open a hot radiator cap. If the temperature gauge is high, stop driving to avoid engine damage.",
    ),
    "Air Leak": (
        [
            "Inspect intake tract, air-box connections and the intake boot for cracks or loose clamps.",
            "Check vacuum hoses, PCV system and brake booster line for splits or disconnections.",
            "Perform a smoke test of the intake system to locate unmetered air entry.",
            "Inspect the intake manifold and throttle body gaskets for leakage.",
            "Verify the MAP sensor reading is consistent with engine load at idle.",
            "Repair or replace leaking components, then clear codes and rescan to confirm normal trims.",
        ],
        "Keep hands and tools clear of the fan and belts when testing with the engine running.",
    ),
}
_GENERIC_FALLBACK = (
    [
        "Read stored DTCs and freeze-frame data for the detected fault.",
        "Visually inspect the affected system for leaks, damage or loose connections.",
        "Verify the relevant sensor readings against OEM specifications.",
        "Repair or replace the faulty component, clear codes and rescan to confirm.",
    ],
    "Follow the manufacturer's service procedure and standard workshop safety practice.",
)

_LIST_PREFIX_RE = re.compile(r"^\s*(?:\d+\s*[.):-]|[-*\u2022])\s*")


@dataclass(frozen=True)
class RepairResult:
    steps: list[str]
    safety_notice: str | None
    source: Literal["gemini", "fallback", "not_required"]


def _fallback_result(label: str) -> RepairResult:
    steps, notice = _FALLBACK_GUIDES.get(label, _GENERIC_FALLBACK)
    return RepairResult(steps=list(steps), safety_notice=notice, source="fallback")


def _sanitize_guide(guide: RepairGuide) -> tuple[list[str], str | None]:
    """Clean model output: strip list markers, cap length/count, mask any PII."""
    steps: list[str] = []
    for raw in guide.steps:
        text = mask_pii(_LIST_PREFIX_RE.sub("", str(raw)).strip())
        if text:
            steps.append(text[:300])
        if len(steps) == 8:
            break
    notice = mask_pii(guide.safety_notice.strip())[:300] if guide.safety_notice else None
    return steps, notice or None


def _build_prompt(req: TelemetryReading, pred: Prediction) -> str:
    # Only numeric telemetry and the fault label are sent. No identifiers, ever.
    map_residual = req.intake_pressure_kpa - (MAP_BASELINE_KPA + MAP_PER_LOAD_PCT * req.engine_load_pct)
    return (
        f"Detected fault: {pred.label} (model confidence {pred.confidence:.2f})\n"
        f"Engine speed: {req.rpm:.0f} rpm\n"
        f"Coolant temperature: {req.coolant_temp_c:.1f} C\n"
        f"Short-term fuel trim: {req.short_fuel_trim_pct:+.1f} %\n"
        f"Long-term fuel trim: {req.long_fuel_trim_pct:+.1f} %\n"
        f"Intake manifold pressure: {req.intake_pressure_kpa:.1f} kPa\n"
        f"Calculated engine load: {req.engine_load_pct:.1f} %\n"
        f"Intake pressure minus load-expected pressure: {map_residual:+.1f} kPa\n"
        "Generate the repair guide."
    )


async def _call_gemini(rt: Runtime, prompt: str) -> RepairGuide:
    assert rt.genai_client is not None and rt.llm_config is not None
    async with rt.llm_semaphore:  # acquisition time counts toward the deadline (see wait_for below)
        response = await rt.genai_client.aio.models.generate_content(
            model=rt.settings.gemini_model, contents=prompt, config=rt.llm_config
        )
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, RepairGuide):
        return parsed
    if not response.text:
        raise ValueError("Empty Gemini response")
    return RepairGuide.model_validate_json(response.text)


async def generate_repair_guide(rt: Runtime, req: TelemetryReading, pred: Prediction) -> RepairResult:
    if rt.genai_client is None or not rt.breaker.allow():
        return _fallback_result(pred.label)

    try:
        guide = await asyncio.wait_for(_call_gemini(rt, _build_prompt(req, pred)),
                                       timeout=rt.settings.llm_timeout_s)
        steps, notice = _sanitize_guide(guide)
        if len(steps) < 2:
            raise ValueError("Gemini returned too few usable steps")
        rt.breaker.record_success()
        return RepairResult(steps=steps, safety_notice=notice, source="gemini")
    except asyncio.TimeoutError:
        rt.breaker.record_failure()
        log_event(logging.WARNING, "Gemini call timed out; using fallback guide",
                  timeout_s=rt.settings.llm_timeout_s)
    except Exception as exc:  # SDK/network/schema errors: degrade, never fail the scan
        rt.breaker.record_failure()
        log_event(logging.WARNING, "Gemini call failed; using fallback guide",
                  error_type=type(exc).__name__)
    return _fallback_result(pred.label)


# =============================================================================
# BigQuery logging (background task, pseudonymised, bounded retries)
# =============================================================================
def pseudonymize(settings: Settings, identifier: str | None) -> str:
    """HMAC-SHA256 pseudonym. Irreversible without the secret salt; never log the input."""
    if not identifier:
        return "anonymous"
    return hmac.new(settings.pii_salt, identifier.encode("utf-8"), hashlib.sha256).hexdigest()[:32]


def build_bq_row(rt: Runtime, req: ScanRequest, pred: Prediction, result: RepairResult) -> dict[str, Any]:
    """Row matching autofix_db.telemetry_logs (see scripts/gcp_setup.sh). Zero raw PII."""
    return {
        "event_timestamp": datetime.now(timezone.utc).isoformat(),
        "vehicle_id": pseudonymize(rt.settings, req.vehicle_id),
        "session_id": req.session_id or uuid.uuid4().hex,
        "rpm": req.rpm,
        "coolant_temp_c": req.coolant_temp_c,
        "short_fuel_trim_pct": req.short_fuel_trim_pct,
        "long_fuel_trim_pct": req.long_fuel_trim_pct,
        "intake_pressure_kpa": req.intake_pressure_kpa,
        "engine_load_pct": req.engine_load_pct,
        "predicted_fault": pred.label,
        "fault_confidence": pred.confidence,
        "model_version": rt.loaded.version,
        "repair_guide": json.dumps(result.steps, ensure_ascii=False) if result.steps else None,
    }


def _insert_row_sync(client: bigquery.Client, table_id: str, row: dict[str, Any],
                     insert_id: str, timeout_s: float) -> None:
    errors = client.insert_rows_json(table_id, [row], row_ids=[insert_id], timeout=timeout_s)
    if errors:
        reasons = sorted({e.get("reason", "unknown") for err in errors for e in err.get("errors", [])})
        raise RuntimeError(f"BigQuery rejected row: {reasons}")  # reasons only, never row content


async def log_telemetry_to_bigquery(rt: Runtime, row: dict[str, Any]) -> None:
    """Runs after the response is sent. Must never raise."""
    if rt.bq_client is None:
        return
    loop = asyncio.get_running_loop()
    insert_id = uuid.uuid4().hex  # same id across retries => BigQuery best-effort dedupe
    settings = rt.settings

    async with rt.bq_semaphore:
        for attempt in range(1, settings.bq_max_attempts + 1):
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(rt.bq_executor, _insert_row_sync, rt.bq_client,
                                         rt.bq_table_id, row, insert_id, settings.bq_timeout_s),
                    timeout=settings.bq_timeout_s + 2,
                )
                return
            except Exception as exc:
                log_event(logging.WARNING, "BigQuery insert failed", attempt=attempt,
                          max_attempts=settings.bq_max_attempts, error_type=type(exc).__name__)
                if attempt < settings.bq_max_attempts:
                    await asyncio.sleep(0.5 * 2 ** (attempt - 1))
    log_event(logging.ERROR, "Telemetry row dropped after retries", table=rt.bq_table_id)


# =============================================================================
# Post-repair verification logic
# =============================================================================
# Healthy bands are calibrated to the synthetic training distribution
# (train_model.py). Replace with OEM specifications for real vehicles.
MAP_BASELINE_KPA = 22.0
MAP_PER_LOAD_PCT = 0.8


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    unit: str
    low: float
    high: float
    extract: Callable[[TelemetryReading], float]

    def deviation(self, value: float) -> float:
        """How far the value sits outside the healthy band (0 when inside)."""
        return max(self.low - value, value - self.high, 0.0)


PARAMETER_SPECS: tuple[ParameterSpec, ...] = (
    ParameterSpec("coolant_temp_c", "C", 80.0, 103.0, lambda r: r.coolant_temp_c),
    ParameterSpec("short_fuel_trim_pct", "%", -10.0, 10.0, lambda r: r.short_fuel_trim_pct),
    ParameterSpec("long_fuel_trim_pct", "%", -10.0, 10.0, lambda r: r.long_fuel_trim_pct),
    # Intake pressure is judged relative to engine load: an air leak shows up as
    # pressure that is too high for the load, not as an absolute value.
    ParameterSpec("intake_pressure_vs_load_kpa", "kPa", -9.0, 9.0,
                  lambda r: r.intake_pressure_kpa - (MAP_BASELINE_KPA + MAP_PER_LOAD_PCT * r.engine_load_pct)),
)


def evaluate_reading(reading: TelemetryReading) -> dict[str, tuple[float, float]]:
    """Return {parameter: (value, deviation)} for every monitored parameter."""
    result: dict[str, tuple[float, float]] = {}
    for spec in PARAMETER_SPECS:
        value = spec.extract(reading)
        result[spec.name] = (round(value, 3), round(spec.deviation(value), 3))
    return result


def verify_normalization(pre: TelemetryReading, post: TelemetryReading) -> tuple[
        float, ResolutionStatus, list[str], list[AnomalyDetail]]:
    pre_eval, post_eval = evaluate_reading(pre), evaluate_reading(post)
    pre_anoms = {n for n, (_, dev) in pre_eval.items() if dev > 0}
    post_anoms = {n for n, (_, dev) in post_eval.items() if dev > 0}
    fixed = pre_anoms - post_anoms
    new = post_anoms - pre_anoms

    if not pre_anoms:
        # Nothing to normalise; the only question is whether the repair broke something.
        pct = 100.0 if not post_anoms else 0.0
        status = ResolutionStatus.NO_PRIOR_ANOMALY if not post_anoms else ResolutionStatus.REGRESSED
    else:
        pct = round(100.0 * len(fixed) / len(pre_anoms), 1)
        if not post_anoms:
            status = ResolutionStatus.RESOLVED
        elif fixed:
            status = ResolutionStatus.PARTIALLY_RESOLVED
        elif new:
            status = ResolutionStatus.REGRESSED
        else:
            status = ResolutionStatus.UNRESOLVED

    remaining: list[AnomalyDetail] = []
    for spec in PARAMETER_SPECS:
        if spec.name not in post_anoms:
            continue
        value, deviation = post_eval[spec.name]
        pre_dev = pre_eval[spec.name][1] if spec.name in pre_anoms else None
        remaining.append(AnomalyDetail(
            parameter=spec.name, unit=spec.unit, observed_value=value,
            healthy_min=spec.low, healthy_max=spec.high, deviation=deviation,
            pre_repair_deviation=pre_dev,
            improved=pre_dev is not None and deviation < pre_dev,
            newly_introduced=spec.name in new,
        ))

    pre_names = [spec.name for spec in PARAMETER_SPECS if spec.name in pre_anoms]
    return pct, status, pre_names, remaining


# =============================================================================
# ASGI middleware: request id, body-size cap, safe headers
# =============================================================================
_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9-]{8,64}")


class RequestContextMiddleware:
    """Pure-ASGI (no BaseHTTPMiddleware overhead). Never trusts client-supplied ids blindly."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        incoming = headers.get("x-request-id", "")
        request_id = incoming if _REQUEST_ID_RE.fullmatch(incoming) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > MAX_BODY_BYTES:  # covers chunked bodies with no Content-Length
                    raise HTTPException(status_code=413, detail="Request body too large")
            return message

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                out = MutableHeaders(scope=message)
                out["X-Request-ID"] = request_id
                out["Cache-Control"] = "no-store"
                out["X-Content-Type-Options"] = "nosniff"
            await send(message)

        try:
            declared = headers.get("content-length", "")
            if declared.isdigit() and int(declared) > MAX_BODY_BYTES:
                response = JSONResponse({"detail": "Request body too large", "request_id": request_id},
                                        status_code=413)
                await response(scope, receive, send_with_headers)
                return
            await self.app(scope, limited_receive, send_with_headers)
        finally:
            request_id_var.reset(token)


# =============================================================================
# Application
# =============================================================================
_IS_PROD = os.getenv("APP_ENV", "development").strip().lower() == "production"

app = FastAPI(
    title="AutoFix AI",
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url=None if _IS_PROD else "/docs",
    redoc_url=None,
    openapi_url=None if _IS_PROD else "/openapi.json",
)
app.add_middleware(RequestContextMiddleware)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 without echoing submitted values (FastAPI's default includes `input`)."""
    detail = [{"loc": list(err.get("loc", ())), "msg": err.get("msg", "Invalid value"),
               "type": err.get("type", "value_error")} for err in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": detail, "request_id": request_id_var.get()})


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled error", extra={"fields": {"path": request.url.path}})
    return JSONResponse(status_code=500,
                        content={"detail": "Internal server error", "request_id": request_id_var.get()})


@app.get("/health", response_model=HealthResponse, status_code=200)
async def health(request: Request) -> HealthResponse:
    rt: Runtime | None = getattr(request.app.state, "runtime", None)
    return HealthResponse(
        status="ok",
        model_loaded=rt is not None,
        llm_available=bool(rt and rt.genai_client is not None),
        bigquery_logging=bool(rt and rt.bq_client is not None),
        version=APP_VERSION,
    )


@app.post("/scan", response_model=ScanResponse)
async def scan(body: ScanRequest, background_tasks: BackgroundTasks, rt: RuntimeDep) -> ScanResponse:
    started = time.perf_counter()

    # 1) CPU inference, off the event loop.
    pred = await classify(rt, body)
    inference_ms = (time.perf_counter() - started) * 1000

    # 2) Repair steps only when a fault is detected.
    is_fault = pred.label != NORMAL_LABEL
    llm_started = time.perf_counter()
    if is_fault:
        result = await generate_repair_guide(rt, body, pred)
    else:
        result = RepairResult(steps=[], safety_notice=None, source="not_required")
    llm_ms = (time.perf_counter() - llm_started) * 1000 if is_fault else 0.0

    # 3) Non-blocking, pseudonymised telemetry logging after the response is sent.
    background_tasks.add_task(log_telemetry_to_bigquery, rt, build_bq_row(rt, body, pred, result))

    log_event(logging.INFO, "scan completed", fault_state=pred.label, confidence=pred.confidence,
              repair_source=result.source, inference_ms=round(inference_ms, 2),
              llm_ms=round(llm_ms, 2), total_ms=round((time.perf_counter() - started) * 1000, 2))

    return ScanResponse(
        request_id=request_id_var.get(),
        fault_state=pred.label,
        confidence=pred.confidence,
        probabilities=pred.probabilities,
        is_fault=is_fault,
        low_confidence=pred.confidence < rt.settings.min_confidence,
        repair_steps=result.steps,
        safety_notice=result.safety_notice,
        repair_source=result.source,
        model_version=rt.loaded.version,
    )


@app.post("/post-repair-check", response_model=PostRepairResponse)
async def post_repair_check(body: PostRepairRequest, rt: RuntimeDep) -> PostRepairResponse:
    # Rule-based normalisation check (pure arithmetic, microseconds: safe inline).
    pct, status, pre_anomalies, remaining = verify_normalization(body.pre_repair, body.post_repair)

    # Independent second opinion from the classifier on the rescan.
    post_pred = await classify(rt, body.post_repair)

    notes: list[str] = []
    if not pre_anomalies:
        notes.append("No monitored parameter was out of range before the repair; nothing to normalise.")
    if abs(body.pre_repair.rpm - body.post_repair.rpm) > 1000 or \
            abs(body.pre_repair.engine_load_pct - body.post_repair.engine_load_pct) > 25:
        notes.append("Operating conditions differ substantially between scans; "
                     "rescan at a comparable RPM and load for a reliable comparison.")
    if post_pred.label != NORMAL_LABEL and post_pred.confidence >= rt.settings.min_confidence \
            and status in (ResolutionStatus.RESOLVED, ResolutionStatus.NO_PRIOR_ANOMALY):
        notes.append(f"Parameters are in range, but the classifier still flags '{post_pred.label}' "
                     f"({post_pred.confidence:.0%}). Recommend a steady-state rescan before sign-off.")
    if any(r.newly_introduced for r in remaining):
        notes.append("The post-repair scan shows anomalies that were not present before the repair.")

    # Counts only: free-text repair actions are never logged (DPDP data minimisation).
    log_event(logging.INFO, "post-repair check completed", resolution_status=status.value,
              normalization_pct=pct, pre_anomaly_count=len(pre_anomalies),
              remaining_anomaly_count=len(remaining), repair_action_count=len(body.repair_actions),
              post_fault_state=post_pred.label)

    return PostRepairResponse(
        request_id=request_id_var.get(),
        normalization_pct=pct,
        resolution_status=status,
        pre_repair_anomalies=pre_anomalies,
        remaining_anomalies=remaining,
        post_repair_classification=PostRepairClassification(fault_state=post_pred.label,
                                                            confidence=post_pred.confidence),
        notes=notes,
        model_version=rt.loaded.version,
    )


if __name__ == "__main__":
    import uvicorn

    # Single worker: the model lives in process memory. Scale with Cloud Run instances.
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")),
                workers=1, access_log=False)
