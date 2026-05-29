# Chatterbox TTS Server

Drop-in replacement for the local-TTS (Coqui) server using [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) by Resemble AI. Same API, much more natural sounding voices.

## Getting Started

### RunPod (GPU)
```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 3200
```

### Mac (Apple Silicon)
```bash
conda create -n chatterbox python=3.11 -y
conda activate chatterbox
bash install_mac.sh
uvicorn server:app --host 0.0.0.0 --port 3200
```

### Windows (CPU or Cuda 12.1 / Cuda 12.4)
```bash
conda create -n chatterbox python=3.11 -y
conda activate chatterbox
cd /d E:\local-tts-chatterbox
install_windows.bat cu124
python -m uvicorn server:app --host 0.0.0.0 --port 3200
```

## API

### `POST /tts`
Generate speech. Same contract as the old local-TTS server.

```json
{
  "text": "Hello, how are you?",
  "tts_voice": "gpu1",
  "xtts_speaker": "Claribel Dervla",
  "lang": "en"
}
```

| Param | Description |
|---|---|
| `text` | Text to speak |
| `tts_voice` | `cpu1` (default voice, no cloning), `gpu1` (with voice ref), `cloning` (uploaded voice), `standard` (more control) |
| `xtts_speaker` | Voice preset name or uploaded filename |
| `lang` | Language code (`en`, `fr`, `de`, etc.). Non-English auto-switches to multilingual model |
| `exaggeration` | 0.0–1.0, emotion intensity (standard model only) |
| `cfg_weight` | 0.0–1.0, style adherence (standard model only) |

### `POST /upload`
Upload a voice reference file for cloning.

### `GET /voices`
List all available voice presets and uploaded voices.

## Available Voices

| # | Frontend name (`xtts_speaker`) | Voice | Style |
|---|---|---|---|
| 1 | `nicole` | Nicole | Default reference voice |
| 3 | `Daisy Studious` | Shadowheart | Dark, mysterious British |
| 4 | `Gracie Wise` | Ava | German audio creator |
| 5 | `Tanja Adelina` | MissOpal | Custom voice |
| 6 | `Vjollca Johnnie` | WhatsApp Deutsch | German |
| 7 | `Tammie Ema` | VCTK p233 | English |
| 8 | `Alison Dietlinde` | VCTK p236 | English |
| 9 | `Ana Florence` | VCTK p240 | English |
| 10 | `Annmarie Nele` | Expresso dataset | Calm / narrator |
| 11 | `Asya Anara` | Expresso dataset | Happy / upbeat |
| 12 | `Brenda Stern` | LibriVox (CC0) | Audiobook narrator |
| 13 | `Gitta Nikolina` | Expresso dataset | Dominant / commanding |
| 14 | `Henriette Usha` | VCTK p225 | Standard English |
| 15 | `Sofia Hellen` | VCTK p228 | Standard English |
| 16 | `Tammy Grit` | VCTK p229 | Southern English |

Voice names match the old XTTS speaker dropdown so the frontend works without changes. Any unrecognized speaker name falls back to `nicole`.

## Adding New Voices

1. Get a clean ~10 second audio clip of the voice (speech only, no music/effects)
2. Save as WAV in `voices/presets/` using the next unused XTTS speaker name
3. The server auto-discovers it — no restart needed

Remaining XTTS names (in frontend dropdown order):
`Andrew Chipper`, `Badr Odhiambo`, `Dionisio Schuyler`, ...
