import os
import time
import gc
from collections import OrderedDict
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import uuid
import torch
import torchaudio
import shutil

# --- Performance: enable TF32 matmul precision where available ---
torch.set_float32_matmul_precision("high")

# Fix for Apple Silicon: resemble-perth watermarker has no ARM binary,
# so PerthImplicitWatermarker is None. Patch it with DummyWatermarker.
# This only affects the inaudible watermark — audio quality is identical.
import perth
if perth.PerthImplicitWatermarker is None:
    print("Patching perth: using DummyWatermarker (no ARM binary available)")
    perth.PerthImplicitWatermarker = perth.DummyWatermarker

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.middleware("http")
async def add_custom_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response

# Device detection
# Note: MPS (Apple Silicon GPU) is disabled by default — Chatterbox has known
# tensor allocation issues with MPS. Set CHATTERBOX_DEVICE=mps to force it.
override = os.environ.get("CHATTERBOX_DEVICE")
if override:
    device = override
elif torch.cuda.is_available():
    device = "cuda"
    print("GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("CUDA Version:", torch.version.cuda)
else:
    device = "cpu"
    print("Using CPU (set CHATTERBOX_DEVICE=mps to try Apple Silicon GPU)")

print(f"Device: {device}")

# Keep only the model in active use (best for memory-constrained / Apple Silicon).
# Set CHATTERBOX_MAX_MODELS to N to keep an LRU of N models resident instead.
MAX_MODELS = int(os.environ.get("CHATTERBOX_MAX_MODELS", "1"))

# Lazy-loaded model cache (LRU-ordered)
models = OrderedDict()

# Voice embedding cache: avoids re-computing speaker embeddings for repeated voices.
# Key = (model_type, abs_path, mtime), Value = model.conds (Conditionals dataclass)
# Bounded LRU so it can't grow without limit.
_voice_cache = OrderedDict()
_VOICE_CACHE_MAX = int(os.environ.get("CHATTERBOX_VOICE_CACHE", "16"))


def _free_memory():
    """Force Python GC + clear the device allocator cache."""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def unload_model(model_type):
    """Fully evict a model and release its memory."""
    model = models.pop(model_type, None)
    if model is None:
        return

    # Critical: drop voice-cache entries that reference this model's tensors,
    # otherwise the memory stays pinned even after the model is gone.
    for key in [k for k in _voice_cache if k[0] == model_type]:
        del _voice_cache[key]

    try:
        model.to("cpu")  # move device tensors off GPU before freeing
    except Exception:
        pass
    del model
    _free_memory()
    print(f"Unloaded {model_type} model")


def _optimize_model(model, model_type):
    """Apply post-load inference optimizations to a Chatterbox model."""
    # LSTM flatten_parameters — makes LSTM memory contiguous for faster CPU inference
    try:
        model.ve.lstm.flatten_parameters()
        print(f"  + LSTM flatten_parameters")
    except Exception as e:
        print(f"  - LSTM flatten_parameters failed: {e}")

    # HiFiGAN: remove weight_norm from vocoder Conv layers (redundant at inference time).
    # The library uses torch.nn.utils.parametrizations.weight_norm (new API), so we
    # remove via parametrize.remove_parametrizations. Skip m_source (no weight_norm).
    try:
        vocoder = model.s3gen.mel2wav
        from torch.nn.utils.parametrize import remove_parametrizations
        count = 0
        for module in vocoder.modules():
            if hasattr(module, "parametrizations") and hasattr(module.parametrizations, "weight"):
                remove_parametrizations(module, "weight")
                count += 1
        print(f"  + HiFiGAN weight_norm removed ({count} layers)")
    except Exception as e:
        print(f"  - HiFiGAN weight_norm removal failed: {e}")

    # Optional: half precision (CHATTERBOX_DTYPE=float16)
    dtype_str = os.environ.get("CHATTERBOX_DTYPE", "")
    if dtype_str == "float16":
        try:
            model.half()
            print(f"  + Converted to float16")
        except Exception as e:
            print(f"  - float16 conversion failed: {e}")

    # Optional: torch.compile on T3 transformer (CHATTERBOX_COMPILE=true)
    if os.environ.get("CHATTERBOX_COMPILE", "").lower() in ("1", "true"):
        try:
            model.t3.tfmr = torch.compile(model.t3.tfmr)
            print(f"  + torch.compile on T3 transformer (first call will be slow)")
        except Exception as e:
            print(f"  - torch.compile failed: {e}")


def get_model(model_type="turbo"):
    """Lazy-load, optimize, and cache Chatterbox models (with LRU eviction)."""
    if model_type in models:
        models.move_to_end(model_type)  # mark as most-recently-used
        return models[model_type]

    # Evict least-recently-used models until there's room for the new one.
    while len(models) >= MAX_MODELS:
        oldest = next(iter(models))
        unload_model(oldest)

    t0 = time.perf_counter()
    if model_type == "turbo":
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        models[model_type] = ChatterboxTurboTTS.from_pretrained(device=device)
    elif model_type == "standard":
        from chatterbox.tts import ChatterboxTTS
        models[model_type] = ChatterboxTTS.from_pretrained(device=device)
    elif model_type == "multilingual":
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        # Multilingual checkpoint was saved with CUDA tensors — patch
        # torch.load to force map_location so it works on CPU/MPS.
        _orig_load = torch.load
        torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "map_location": device})
        try:
            models[model_type] = ChatterboxMultilingualTTS.from_pretrained(device=device)
        finally:
            torch.load = _orig_load
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    load_time = time.perf_counter() - t0
    print(f"Loaded Chatterbox {model_type} model on {device} ({load_time:.1f}s)")
    _optimize_model(models[model_type], model_type)
    return models[model_type]


def _prepare_voice(model, model_type, audio_prompt_path):
    """Prepare voice conditionals with caching. Returns True if conditionals are ready."""
    if not audio_prompt_path:
        return False

    abs_path = os.path.abspath(audio_prompt_path)
    try:
        mtime = os.path.getmtime(abs_path)
    except OSError:
        return False

    cache_key = (model_type, abs_path, mtime)

    if cache_key in _voice_cache:
        model.conds = _voice_cache[cache_key]
        _voice_cache.move_to_end(cache_key)  # mark as most-recently-used
        print(f"  Voice cache HIT: {os.path.basename(abs_path)}")
        return True

    # Cache miss — compute conditionals (expensive: loads WAV, runs LSTM + CAMPPlus + S3Tokenizer)
    t0 = time.perf_counter()
    model.prepare_conditionals(abs_path)
    prep_time = time.perf_counter() - t0
    _voice_cache[cache_key] = model.conds
    _voice_cache.move_to_end(cache_key)

    # Bound the voice cache (drop oldest entries).
    while len(_voice_cache) > _VOICE_CACHE_MAX:
        _voice_cache.popitem(last=False)

    print(f"  Voice cache MISS: {os.path.basename(abs_path)} (computed in {prep_time:.2f}s, cached)")
    return True


VOICES_DIR = "./voices"
PRESETS_DIR = "./voices/presets"
MY_VOICES_DIR = "./voices/my_voices"


def resolve_audio_prompt(tts_voice, xtts_speaker):
    """Resolve the voice reference audio path from the request parameters."""
    if tts_voice == "gpu1" or tts_voice == "standard":
        if xtts_speaker == "nicole" or xtts_speaker is None:
            return f"{VOICES_DIR}/nicole.wav"
        # Check presets first, then my_voices
        for directory in [PRESETS_DIR, MY_VOICES_DIR]:
            # Try exact match, then with .wav extension
            for name in [xtts_speaker, f"{xtts_speaker}.wav"]:
                path = os.path.join(directory, name)
                if os.path.exists(path):
                    return path
        return f"{VOICES_DIR}/nicole.wav"

    elif tts_voice == "cloning":
        if xtts_speaker:
            # Check my_voices first, then presets
            for directory in [MY_VOICES_DIR, PRESETS_DIR]:
                for name in [xtts_speaker, f"{xtts_speaker}.wav"]:
                    path = os.path.join(directory, name)
                    if os.path.exists(path):
                        return path
            return f"{MY_VOICES_DIR}/{xtts_speaker}"

    return None


@app.post("/tts")
async def generate_tts(request: Request):
    data = await request.json()
    text = data.get("text", "").strip()
    lang = data.get("lang", "en")
    xtts_speaker = data.get("xtts_speaker")
    tts_voice = data.get("tts_voice", "gpu1")

    # Chatterbox-specific optional params
    exaggeration = data.get("exaggeration", 0.5)
    cfg_weight = data.get("cfg_weight", 0.5)

    if not text:
        return JSONResponse(content={"error": "No text provided"}, status_code=400)

    audio_prompt_path = resolve_audio_prompt(tts_voice, xtts_speaker)

    # Validate audio prompt exists
    if audio_prompt_path and not os.path.exists(audio_prompt_path):
        print(f"Warning: audio prompt not found: {audio_prompt_path}")
        audio_prompt_path = None

    # Select model: multilingual for non-English, otherwise turbo or standard
    if lang and lang != "en":
        model_type = "multilingual"
        model = get_model(model_type)
        voice_cached = _prepare_voice(model, model_type, audio_prompt_path)
        generate_kwargs = {
            "text": text,
            "language_id": lang,
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
        }
        if audio_prompt_path and not voice_cached:
            generate_kwargs["audio_prompt_path"] = audio_prompt_path

    elif tts_voice == "standard":
        # Explicit standard model request — supports exaggeration/cfg_weight
        model_type = "standard"
        model = get_model(model_type)
        voice_cached = _prepare_voice(model, model_type, audio_prompt_path)
        generate_kwargs = {
            "text": text,
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
        }
        if audio_prompt_path and not voice_cached:
            generate_kwargs["audio_prompt_path"] = audio_prompt_path

    else:
        # cpu1, gpu1, cloning, turbo → use Turbo (fastest, supports [laugh] etc.)
        model_type = "turbo"
        model = get_model(model_type)
        voice_cached = _prepare_voice(model, model_type, audio_prompt_path)
        generate_kwargs = {"text": text}
        if audio_prompt_path and not voice_cached:
            generate_kwargs["audio_prompt_path"] = audio_prompt_path

    print(f"Generating: tts_voice={tts_voice}, lang={lang}, speaker={xtts_speaker}")
    print(f"Model kwargs: { {k: v for k, v in generate_kwargs.items() if k != 'text'} }")

    t0 = time.perf_counter()
    with torch.inference_mode():
        wav = model.generate(**generate_kwargs)
    gen_time = time.perf_counter() - t0
    print(f"Generated in {gen_time:.2f}s ({len(text)} chars)")

    tmp_wav = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.wav")
    torchaudio.save(tmp_wav, wav, model.sr)

    return FileResponse(tmp_wav, media_type="audio/wav", filename="speech.wav")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    print(file.filename)
    file_location = f"./voices/my_voices/{file.filename}"

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return JSONResponse(content={"filename": file.filename, "success": True})


@app.get("/voices")
async def list_voices():
    """List all available voice presets and uploaded voices."""
    presets = []
    if os.path.exists(PRESETS_DIR):
        presets = sorted([
            f for f in os.listdir(PRESETS_DIR)
            if f.endswith((".wav", ".mp3")) and not f.startswith(".")
        ])

    my_voices = []
    if os.path.exists(MY_VOICES_DIR):
        my_voices = sorted([
            f for f in os.listdir(MY_VOICES_DIR)
            if f.endswith((".wav", ".mp3")) and not f.startswith(".")
        ])

    return JSONResponse(content={
        "presets": presets,
        "my_voices": my_voices,
        "default": "nicole.wav",
    })


@app.on_event("startup")
async def preload_models():
    """Pre-load models at startup. Control via CHATTERBOX_PRELOAD env var (default: turbo)."""
    preload = os.environ.get("CHATTERBOX_PRELOAD", "turbo")
    if preload:
        requested = [m.strip() for m in preload.split(",") if m.strip()]
        # Warn if you're asking to preload more models than will fit — they'd
        # just evict each other immediately at startup.
        if len(requested) > MAX_MODELS:
            print(f"Warning: CHATTERBOX_PRELOAD lists {len(requested)} models but "
                  f"CHATTERBOX_MAX_MODELS={MAX_MODELS}. Only the last {MAX_MODELS} "
                  f"will remain resident.")
        for model_type in requested:
            print(f"Pre-loading {model_type} model...")
            get_model(model_type)