# Training und Serving trennen — `training-atr-models`

Stand 16.09.2026. Grundlage ist eine Bestandsaufnahme der tatsächlichen
Kopplung, nicht eine Schätzung; jede Zeilenangabe unten ist nachgeprüft.

---

## 0. Was die Trennung wirklich kostet

Die verbreitete Annahme wäre: „die beiden Hälften sind verwoben". Sie sind es
**im Code kaum**. Nachgezählt:

| Richtung | Umfang |
|---|---|
| `training/` → Serving | **ein** Modul: `atr_serving.registry`, an zwei Stellen (`training/overlay.py:26`, `training/base_models.py:36`) plus vier Engine-Runner |
| Serving → `training/` | **drei** Stellen: `app.py:16` (`overlay`), `api/train_routes.py:26-27` (`backends`, `contracts`) |

Das ist alles. `TrainerSettings` ist bereits eine eigenständige `BaseSettings`
mit eigenem Präfix (`ATR_TRAIN_`), und `training/` importiert weder `config.py`
noch `manager.py`, `pipeline.py`, `clients.py` oder `image_io.py`.

**Die eigentliche Kopplung ist das Dateisystem**, und die ist massiv:

| Was | Wo | Bricht bei Trennung |
|---|---|---|
| `models.local.yaml` (Overlay) | Trainer schreibt (`training/settings.py:39`), Gateway liest beim Start (`app.py:42`) | **Ja** — zwei Prozesse, eine Datei |
| `trained_root` + `local_path` | Trainer kopiert Gewichte hin, Gateway legt den **absoluten Pfad** in `ModelSpec.local_path`, kraken öffnet ihn (`kraken_loader.py:56`) | **Ja** — jeder Pfad zeigt ins Leere |
| `vllm_merged_dir` | `scripts/merge_loras.py:401` schreibt, `manager.py:85` liest | **Ja** |
| HF-Cache | `~/.cache/huggingface/hub` → Symlink auf die CIFS-Freigabe | nur wenn der neue Server sie nicht mountet |
| `REPO_ROOT` | beide Hälften nehmen **einen** Checkout an (`training/settings.py:24`, `config.py`) | **Ja**, per Definition |

Und zwei HTTP-Kanten, die heute Loopback sind:

- `train_url = http://127.0.0.1:8204` (`config.py:61`) — der `/train/*`-Proxy
- `gateway_url = http://127.0.0.1:8200` (`training/settings.py:101`) — das
  Promotion-Gate postet eine Seite an `/ocr`

Beide sind schon HTTP. **Sie sind die Naht.** Die Trennung besteht im Kern
darin, aus zwei Loopback-Kanten zwei Netzwerkkanten zu machen und alles, was
heute über eine gemeinsame Platte läuft, ebenfalls über diese Kanten zu führen.

---

## 1. Zielarchitektur

```
   tei.dh.unibe.ch                asterAIx (serving)            <neuer Server> (training)
   ┌──────────────────┐           ┌────────────────────┐        ┌──────────────────────┐
   │ agentic_historian│──:8200───▶│ atr-gateway        │─:8204─▶│ atr-train            │
   │  atr_status.py   │  X-API-Key│  /recognize /ocr   │  mTLS  │  /jobs /gpu /health  │
   │  atr_watch.py    │           │  /train/*  (Proxy) │  o. VPN│  /gpu (nvidia-smi)   │
   │  atr_batch.py    │           │  /models           │        │                      │
   └──────────────────┘           │  /admin/register   │◀───────│  register-Stage      │
                                  └────────────────────┘  HTTP  └──────────────────────┘
                                     kraken/trocr/party/vLLM        kraken/trocr/vlm
                                     GPU 1 nur noch Serving          eigene GPU(s)
```

**Unverändert bleibt**, und das ist die Vorgabe:

- Der Bot spricht weiter **ausschliesslich** `:8200` an. `atr_status.py:12-15`
  hält fest, warum: asterAIx bindet den Trainer an `127.0.0.1` und öffnet nur
  `:8200` zu tei. Nach der Trennung zeigt derselbe Proxy über das Netz — der Bot
  merkt nichts, weder in `atr_status.py` noch in `atr_watch.py` noch in der
  Konfiguration.
- Alle acht `/train/*`-Routen, Pfade und Antwortformate.
- Die fünf Stages, die Job-IDs, `job.json`, die Log-Endpunkte.

**Neu ist** eine dritte Route am Gateway, `POST /admin/register`, über die der
Trainer ein fertiges Modell anmeldet, statt eine Datei zu schreiben, die beide
sehen.

---

## 2. Entscheidungen, die vorab zu treffen sind

Diese vier ändern den Plan, nicht nur seine Ausführung.

### E1 — Mountet der neue Server die CIFS-Freigabe?

Davon hängt der teuerste Teil ab.

- **Ja**: `jobs_root` und der HF-Cache bleiben, wo sie sind. Die
  Trainingsdaten (6,6 TB) müssen nicht zweimal existieren, und der Handover der
  Gewichte kann weiterhin über `trained_root` laufen — der Gateway liest dann
  einen Pfad, den beide sehen. Aufwand: klein.
- **Nein**: Datasets werden auf dem neuen Server neu vom Hub gezogen (der
  HF-Cache liegt heute auf der Freigabe und wird mit `lassberg/vlm_training`
  geteilt), und die Gewichte müssen über HTTP zum Gateway. Aufwand: Epic T3
  wächst deutlich.

### E2 — Welcher Server, und wie erreicht ihn asterAIx?

Der Trainer bindet heute auf `127.0.0.1`. Über das Netz braucht es eine
Authentifizierung, die nicht „niemand kommt dran" heisst. Vorschlag:
derselbe `X-API-Key`-Mechanismus wie am Gateway, plus Bindung an das
VPN-Interface. mTLS wäre sauberer und ist mehr Arbeit.

### E3 — Wird die gemeinsame Vokabel ein Paket, oder verschwindet sie?

`ModelSpec` (Serving) und `TrainRequest` (Training) sind heute zwei parallele
pydantic-Vokabulare, die sich **per Konvention** einig sind — sie teilen keinen
einzigen Import. `registry.py:16` kennt vier Engines, `contracts.py:103` drei;
nichts erzwingt den Abgleich.

- **Variante A — geteiltes Paket** (`atr-model-contracts`): korrekt, aber ein
  drittes Repo, das bei jeder Änderung in beiden Versionen hängt.
- **Variante B — die Naht wird dünn genug, dass es keins braucht** (empfohlen):
  der Proxy hört auf, `TrainRequest` ein zweites Mal zu validieren
  (`train_routes.py:85`), und leitet den Body durch; `SUPPORTED_ENGINES`
  (`train_routes.py:37`) kommt aus `/health` des Trainers statt aus
  `training.backends`. Damit fallen **beide** Serving→Training-Importe im
  Proxy weg. Der dritte (`app.py:16`, Overlay) fällt mit Epic T3.

Variante B bedeutet: der Trainer validiert seine eigenen Anfragen, und der
Gateway reicht Fehler durch — was `TrainerError` (`clients.py:245`) heute schon
kann, es hält den Status des Trainers durch statt ihn auf 502 zu plätten.

### E4 — Was passiert mit `eval/` und den 13 `scripts/`?

`eval/` ist Serving-seitig, importiert aber `atr_serving.training.textmetrics`.
Dreizehn Skripte (`merge_loras`, `publish_to_hub`, `plan_corpus`,
`stratified_eval_set`, `compare_eval_reports`, `audit_eval_material`, …) hängen
an `atr_serving.training`. Die meisten gehören klar ins Trainingsrepo. Zwei
Ausnahmen brauchen eine Entscheidung:

- `scripts/merge_loras.py` **spreizt beide Hälften** und beide Settings-Klassen
  (`:37-40`, `:388`, `:392`). Es macht aus einem Adapter ein servierbares
  Modell — das ist Serving-Arbeit auf Training-Output.
- `textmetrics` (CER/WER samt der invertierten Edit-Konvention) wird von beiden
  gebraucht. Kandidat für das geteilte Paket, falls E3 → A.

---

## 3. Epics

### T1 — `training-atr-models` existiert und ist grün

Das Repo anlegen und den Trainingscode hineinbewegen, **ohne** Verhaltens-
änderung. Am Ende läuft die Trainingssuite dort, die Servingsuite hier, und
beide sind grün.

| Issue | Inhalt |
|---|---|
| T1.1 | Repo anlegen, `pyproject.toml`, CI (dieselbe Matrix wie hier), `.env.example` |
| T1.2 | `src/atr_serving/training/` → `src/atr_training/` verschieben, samt `engines/{kraken,trocr,vlm}_train_svc/` |
| T1.3 | Die 44 trainingsseitigen Testdateien mitnehmen, `tests/conftest.py` (Artefakt-Cache-Rail) ebenfalls |
| T1.4 | `deploy/systemd/atr-train.service`, `docs/DEPLOY.md`-Trainingsteil, `.venvs/{kraken,trocr,vlm}-train` |
| T1.5 | Die trainingsseitigen `scripts/` und `ubelix/` mitnehmen (E4) |
| T1.6 | `atr_serving.registry`-Abhängigkeit auflösen: `base_models.py` und `overlay.py` (siehe T3) |

**Tests**: die bestehenden 44 Dateien müssen ohne Änderung ihrer Zusicherungen
durchlaufen. Ein Import von `atr_serving.*` im neuen Repo ist ein Fehler — ein
Test, der das Paket in `sys.modules` verbietet, hält das fest.

### T2 — Der Proxy überquert das Netz

| Issue | Inhalt |
|---|---|
| T2.1 | `train_url` darf ein entfernter Host sein; Timeouts, Retries und ein ehrlicher 504 statt eines hängenden Requests |
| T2.2 | Authentifizierung zum Trainer (E2) — `X-API-Key` auf allen `/jobs*`-Routen des Trainers, heute ungeschützt, weil Loopback |
| T2.3 | `SUPPORTED_ENGINES` aus `/health` des Trainers statt aus `training.backends` (E3-B) |
| T2.4 | `TrainRequest.model_validate` aus dem Proxy entfernen; der Trainer validiert, der Proxy reicht durch (E3-B) |
| T2.5 | **`/train/gpu` ist heute hybrid** (`train_routes.py:129-172`): es liest lokal `nvidia-smi` und mischt die Jobliste des Trainers dazu. Nach der Trennung ist „lokal" der falsche Rechner. Die Route muss vollständig zum Trainer proxen, der die Karten hat |
| T2.6 | `/health` des Gateways: `service_urls()` (`config.py:76`) führt den Trainer als Dienst; über Netz braucht das ein eigenes Timeout, damit ein langsamer Trainingsserver nicht die Health-Antwort blockiert |

**Tests**: `test_train_proxy_routes.py` bleibt, verliert aber seine
Training-Importe — es baut die Gateway-App gegen einen **gefakten** Trainer,
nicht gegen den echten. Neu: ein Test, dass ein nicht erreichbarer Trainer 503
mit Begründung liefert und nicht 500; einer, dass `/train/gpu` die Karten des
**Trainings**servers meldet.

### T3 — Der Handover wird eine API statt einer Datei

Der teuerste Epic, und der, der die Trennung erst echt macht.

| Issue | Inhalt |
|---|---|
| T3.1 | `POST /admin/register` am Gateway: nimmt `ModelSpec`-Felder plus die Gewichte-Herkunft, schreibt das Overlay **gatewayseitig** |
| T3.2 | Die drei `_register`-Implementierungen rufen die Route, statt `upsert_entry` auf eine geteilte Datei zu schreiben |
| T3.3 | Transport der Gewichte (E1): geteilte Freigabe → nur der Pfad wandert; sonst Upload oder Pull über HTTP |
| T3.4 | Das Promotion-Gate (`promote.py:87`) bleibt HTTP und funktioniert unverändert — aber `set_enabled` (`kraken_train_svc/runner.py:415`) schreibt heute wieder die geteilte Datei. Wird Teil von T3.1 |
| T3.5 | `scripts/merge_loras.py` entflechten (E4): der Merge gehört auf die Serving-Seite, ausgelöst durch die Registrierung eines Adapters |
| T3.6 | `app.py:16` (`from atr_serving.training.overlay import …`) fällt weg — das Overlay-Modul wandert ins Serving-Repo, weil nur noch der Gateway es schreibt |

**Tests**: ein Ende-zu-Ende-Test, der einen Trainingslauf mit gefakten Stages
bis `register` führt, die HTTP-Registrierung gegen eine Gateway-Testinstanz
laufen lässt und prüft, dass `/models` das Modell **disabled** zeigt und nach
dem Gate **enabled**. Dazu ein Test, dass eine fehlgeschlagene Registrierung den
Job scheitern lässt, statt ein Modell zu verlieren.

### T4 — Was nach der Trennung überflüssig ist

Ehrlich zu benennen, weil es heute erst gebaut wurde:

| Issue | Inhalt |
|---|---|
| T4.1 | `/gpu-claim` (`kraken_train_svc/app.py:374`), `ModelManager._refuse_while_training`, `GpuBusyError`, `/admin/release-gpu` und `gpu_release.py` **entfernen** — sie koordinieren zwei Prozesse um **eine** Karte, und nach der Trennung gibt es die geteilte Karte nicht mehr |
| T4.2 | `vllm_vram_budget_mb` und die LRU-Verdrängung bleiben, aber gegen eine Karte, die dem Serving allein gehört — die Budgets sind neu zu setzen |
| T4.3 | Die VRAM-Vorprüfung des Trainers (`preflight.py`) bleibt und wird wichtiger, weil sie die einzige verbleibende Karte-belegt-Prüfung ist |

Diese Entfernung erst **nach** dem Cutover, nicht davor: solange beide auf
asterAIx laufen, ist die Sperre das, was einen Lauf vor einer Inferenzanfrage
schützt.

### T5 — Cutover

| Issue | Inhalt |
|---|---|
| T5.1 | Parallelbetrieb: `atr-train` auf dem neuen Server hochziehen, Gateway zeigt noch auf `127.0.0.1` |
| T5.2 | Ein Smoke-Job (`trocr-thun-smoke`) über die neue Kante, alle fünf Stages, Registrierung inklusive |
| T5.3 | `ATR_TRAIN_URL` umstellen, Gateway neu starten, `/train/jobs` aus dem Bot heraus prüfen |
| T5.4 | Der alte Trainer auf asterAIx wird gestoppt, nicht gelöscht — die Job-Historie (44 Records) bleibt lesbar, bis sie migriert ist |
| T5.5 | Job-Historie migrieren oder bewusst abschneiden (Entscheidung) |

---

## 4. Teststrategie

Die Suite zerfällt heute in 44 trainingsseitige, 13 servingseitige und **acht
übergreifende** Dateien. Die acht sind die eigentliche Arbeit:

| Datei | Warum übergreifend | Danach |
|---|---|---|
| `test_train_proxy_routes.py` | baut die Gateway-App, braucht `training.backends`/`contracts` | gegen einen gefakten Trainer, kein Training-Import |
| `test_train_gpu_inspection.py` | `/train/gpu` mischt lokales `nvidia-smi` mit der Jobliste | wandert mit T2.5 zum Trainingsrepo |
| `test_manager.py` | fakt `/gpu-claim` | fällt mit T4.1 grösstenteils weg |
| `test_training_overlay.py` | `registry.ModelSpec` + `training.overlay` | Serving-Repo (Overlay wird gatewayseitig) |
| `test_training_promote.py` | Gate über HTTP | bleibt trainingsseitig, Gateway gefakt |
| `test_training_base_models.py` | liest das echte `config/models.yaml` | Trainingsrepo, mit einer Kopie oder über `/models` |
| `test_train_svc_api.py` | importiert `atr_serving.registry` an einer Stelle | Import auflösen |
| `test_merge_loras_preflight.py` | Skript spreizt beide Hälften | folgt der Entscheidung aus E4 |

**Drei neue Testarten**, die es heute nicht gibt und die die Trennung braucht:

1. **Kontrakttests an der Naht.** Beide Repos halten dieselbe Beispielsammlung
   von `/train/*`-Anfragen und -Antworten vor; der Trainer prüft, dass er sie
   beantwortet, der Gateway, dass er sie erzeugt und versteht. Ohne das ist
   E3-B (keine geteilten Typen) ein Versprechen ohne Prüfung.
2. **Ein Isolationstest je Repo**: ein Import des jeweils anderen Pakets lässt
   die Suite scheitern. Das ist die einzige Zusicherung, die verhindert, dass
   die Naht über Monate wieder zuwächst.
3. **Ein Netzwerkfehler-Test je Route**: Trainer nicht erreichbar, Trainer
   langsam, Trainer antwortet 500. Heute unnötig (Loopback), danach der
   häufigste Fehlerfall.

---

## 5. Reihenfolge

```
E1–E4 entscheiden
   └─ T1 (Repo, grün, noch ohne Netz)
        └─ T2 (Proxy über Netz)   ──┐
        └─ T3 (Handover als API)  ──┴─ parallel möglich
             └─ T5 (Cutover)
                  └─ T4 (Rückbau der GPU-Koordination)
```

T4 **zuletzt**. Solange beide auf einer Karte laufen, ist die heute gebaute
Sperre das, was einen 24-Stunden-Lauf vor einer Inferenzanfrage schützt.

---

## 6. Was dieser Plan nicht behandelt

- **Wie viele GPUs der neue Server hat.** Die Hyperparameter (`batch_size: 1`,
  `accumulate_grad_batches: 16`, `max_seq_len: 4096`) sind auf eine A40 mit
  46 GB abgestimmt, von denen heute 14,4 GB an Serving-Diensten hängen. Auf
  einer Karte, die dem Training allein gehört, sind sie neu zu bestimmen — das
  ist eine eigene Messung, kein Migrationsschritt.
- **UBELIX.** `ubelix/submit_job.py` und `fanout.py` hängen an `jobs_root` und
  wandern mit T1.5, aber die Frage, ob der neue Server UBELIX ersetzt oder
  ergänzt, ist offen.
- **Die Datenlage.** #98, #125 und #135 sind Korrekturen an dem, was trainiert
  wird, nicht daran, wo. Sie laufen unabhängig weiter.
