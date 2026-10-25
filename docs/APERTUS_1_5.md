# Apertus 1.5 als VLM auf idhefix

`swiss-ai/Apertus-v1.5-8B` (Okt. 2026) ist das erste multimodale Apertus: Bild,
Audio und Text rein, Text raus. Es ist ein **allgemeines Instruktionsmodell, kein
HTR-Fine-Tune**. Ziel des Tests ist eine erste Einschätzung: Liest es unsere Seiten
überhaupt, und taugt es als Basis für ein Fine-Tune?

Registry-ID: **`apertus-1.5-8b`** (`config/models.yaml`), `level: page`, lazy auf Karte 1.

## Was man vorher wissen sollte

- **Eigener vLLM.** Weder upstream-vLLM noch upstream-transformers kennen
  `apertus1p5`. swiss-ai liefert Forks (vLLM `a601a9d`, transformers `3797303`).
  Ihr Docker-Image ist ein **CUDA-13-Build und läuft auf Treiber 565 nicht** — dieselbe
  Wand wie bei den Standard-vLLM-Wheels (#132). Darum `.venvs/vllm-apertus`: der Fork
  ist reines Python über upstream-Commit `ae6170f`, und dessen kompilierte Kernels gibt
  es als **cu129**-Nightly. Das ist derselbe Weg, auf dem `vllm-next` läuft.
- **Bildeingang.** Bilder gehen durch einen *diskreten* Vision-Tokenizer
  (VQ-Codebook, 16 px pro Token), keinen ViT-Encoder wie bei Qwen3-VL. Das Modell
  verarbeitet maximal 1.96 MP (≈ 7 700 Bild-Tokens). Die Model Card nennt DocVQA und
  OmniDocBench, aber nichts zu Handschrift. Erwartung deshalb: Druck und saubere
  Schrift eher ja, Kurrent eher schwach. Genau das soll der Test zeigen.
- **Gated.** Die Gewichte sind „auto-approved gated“. Der HF-Token auf idhefix muss
  die Bedingungen auf https://huggingface.co/swiss-ai/Apertus-v1.5-8B akzeptiert haben.
- **Speicher.** ≈ 18.4 GB Gewichte (17.2 GB bf16-LM + 1.2 GB fp32-Tokenizer),
  `vram_mb: 18000`, noch nicht gemessen. Passt auf Karte 1, verdrängt aber beim
  Kaltstart das jeweils residente 8B-Modell.
- **Startflag.** Die Model Card verlangt `--chat-template-content-format string`. Das
  setzt der Registry-Eintrag über das neue Feld `vllm_args`.

## Vorbereitung (heute Abend oder morgen früh, ca. 20–30 min)

```bash
cd ~/Repo/serving-atr-inference && git pull
# 1. Venv bauen: torch 2.11 cu129 + Fork mit vorkompilierten cu129-Kernels.
#    Endet mit "torch 2.11.0+cu129 cuda 12.9". Bricht mit Exit 1 ab, falls ein
#    CUDA-13-torch hereingerutscht ist.
bash scripts/make_venvs.sh vllm-apertus
.venvs/vllm-apertus/bin/python -c "import vllm, transformers; print(vllm.__version__, transformers.__version__)"

# 2. Gewichte holen (~19 GB). Vorher die Gating-Bedingungen akzeptieren.
.venvs/vllm-apertus/bin/python -c "from huggingface_hub import snapshot_download as d; print(d('swiss-ai/Apertus-v1.5-8B'))"

# 3. Gateway neu starten, damit es den neuen Registry-Eintrag liest
systemctl --user restart atr-gateway
curl -s localhost:8200/models | grep -o '"apertus-1.5-8b"'
```

Falls Schritt 1 scheitert: Ohne Docker gibt es auf diesem Treiber keinen Plan B für
den vLLM-Pfad. Dann bleibt der Fallback über transformers (siehe unten).

## Demo

**Eine Seite, freier Prompt** (erster Aufruf = Kaltstart, 1–3 min):

```bash
.venvs/gateway/bin/python scripts/apertus_demo.py seite.jpg
.venvs/gateway/bin/python scripts/apertus_demo.py seite.jpg \
    --prompt "Beschreibe dieses Dokument: Art, Datierung, Schrift, Sprache."
.venvs/gateway/bin/python scripts/apertus_demo.py seite.jpg \
    --prompt "Welche Personen und Orte werden genannt? Antworte als JSON." --thinking
```

Das Skript meldet Laufzeit, Tokens und `finish_reason`. `length` heisst: Die Lesung
bricht mitten im Text ab und ist nicht vollständig.

**Ein kleiner Batch-Vergleich** über den ATR-TEI-MCP (wie bei jedem neuen Modell):
1. `start_batch` mit `models: ["apertus-1.5-8b", "qwen3.5-4b-german-kurrent-xvi-xix-page-v1"]`,
   zuerst `dry_run`, dann `sample: 3` (zufällig, deterministisch; `limit` nimmt
   aufeinanderfolgende Seiten *eines* Dokuments). Mit `keys_from` auf eine
   Ground-Truth-Liste liest der Lauf nur Seiten, die man auch bewerten kann;
2. `compare_readings` auf beide Läufe. Die Zahl dort misst **Uneinigkeit, nicht
   Qualität**;
3. wo Ground Truth existiert, `score_ground_truth`. Erst das ergibt eine CER.

Vorschlag für die Seiten: je 3 aus einem gedruckten Bestand, einem Kurrent-Bestand
des 19. Jh. und einem frühneuzeitlichen Bestand. Dann sieht man, wo die Grenze liegt.

## Was sich morgen zeigen lässt, ehrlich eingeordnet

| Frage | Wie | Erwartung |
|---|---|---|
| Liest es Druck/Fraktur? | Demo + Batch | plausibel |
| Liest es Kurrent/frühneuzeitliche Hände? | Batch gegen das Kurrent-Page-Modell | fraglich (Basismodell, VQ-Tokenizer) |
| Versteht es Dokumente (Typ, Datierung, Regest)? | Demo mit freiem Prompt | sein eigentlicher Vorteil gegenüber den HTR-Modellen |
| Eignet es sich als Fine-Tune-Basis? | erst nach dem Test; Trainer kennt die Architektur noch nicht | offen |

## Fallback: transformers statt vLLM

Falls der vLLM-Fork nicht startet, läuft das Modell auch direkt mit dem
transformers-Fork. Das Snippet steht in der Model Card. Langsam und ohne Gateway, für
eine Demo aber ausreichend:

```bash
CUDA_VISIBLE_DEVICES=1 .venvs/vllm-apertus/bin/python - <<'EOF'
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor
M = "swiss-ai/Apertus-v1.5-8B"
proc = AutoProcessor.from_pretrained(M)
model = AutoModelForMultimodalLM.from_pretrained(M, dtype="auto", device_map="auto").eval()
msgs = [{"role": "user", "content": [{"type": "image", "url": "seite.jpg"},
        {"type": "text", "text": "Transcribe the text in this image."}]}]
inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                               return_dict=True, return_tensors="pt").to(model.device)
with torch.inference_mode():
    out = model.generate(**inp, max_new_tokens=2048)
print(proc.decode(out[0, inp["input_ids"].shape[-1]:], skip_special_tokens=True))
EOF
```

Vorher Karte 1 freimachen. Dieses Skript läuft am Gateway vorbei, dessen VRAM-Buchhaltung
sieht es also nicht.
