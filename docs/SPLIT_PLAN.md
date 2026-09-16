# Training und Serving trennen — `training-atr-models`

Zweite Fassung, 16.09.2026. Die erste war an sechs Stellen sachlich falsch; sie
stehen in §0, weil ein Plan, der seine eigenen Irrtümer verschweigt, sie
weitergibt.

Grundlage: eine Bestandsaufnahme beider Repos und eine Vermessung beider
Maschinen. Jede Zeilenangabe ist nachgeprüft.

---

## 0. Was die erste Fassung falsch hatte

| Behauptung (1. Fassung) | Wirklich |
|---|---|
| „13 Skripte importieren `atr_serving.training`" | **Neun.** `make_split.py`, `smoke_trocr.py`, `check_env.sh` ziehen aus anderen Gründen um — ohne dass vorher ein Import aufzulösen wäre |
| „`merge_loras.py` spreizt beide Settings-Klassen" | Nur **eine**: `atr_serving.config.get_settings` (`:38`). `TrainerSettings` kommt darin nicht vor. Sobald `overlay.py` serving-seitig ist, hat das Skript **null** Training-Importe |
| „`eval/` ist serving-seitig mit einem lästigen Training-Import" | `eval/` hat **null** Serving-Importe. Seine Eingabe ist der Artefaktbaum des Trainers. D4 ist durch den Code gedeckt, nicht dagegen |
| „`textmetrics` wird von beiden Hälften gebraucht → Kandidat für ein geteiltes Paket" | Sein einziger Nicht-Training-Konsument **ist** `eval/`. Zieht `eval/` um, hat `textmetrics` keinen Serving-Konsumenten mehr — die Paketfrage entfällt. **D3 und D4 stützen sich gegenseitig** |
| „44 trainingsseitige, 13 servingseitige, 8 übergreifende Testdateien" | 72 Dateien: **43** training, **16** serving, **4** Grenzgänger, 2 eval, 6 skriptgetrieben, 1 conftest. Und alle vier Grenzgänger kreuzen über **dasselbe eine Symbol** (`atr_serving.registry`), nicht über acht Kopplungen. Die Grenzarbeit ist etwa halb so gross wie veranschlagt |
| „Zwei HTTP-Kanten sind die Naht" | **Drei.** `eval/run_eval.py:44` postet an `/recognize` — nach D4 eine dritte Kante, die dieselbe Firewall-Regel und denselben Schlüssel braucht |

Dazu ein Zitatfehler: `set_enabled` steht in `kraken_train_svc/runner.py:459`, nicht `:415`.

---

## 1. Die beiden Maschinen, gemessen

| | **idhefix** — Serving | **asteraix** — Training |
|---|---|---|
| Adresse | 130.92.59.240 (`srv`, dhserver02) | 130.92.59.242 (`dhserver03`) |
| GPUs | 2 × A40 46 GB; GPU 1 trägt **dauerhaft 14,4 GB** Serving-Dienste (16.09., 21:50: **15 830 MiB** — sie wachsen) | **2 × A40 46 GB, beide 0 MiB belegt** |
| Verbund | — | **NVLink, 4 Links à 14,06 GB/s, P2P OK** |
| CPU / RAM | 48 Kerne / 251 GB | 48 Kerne / 251 GB |
| OS / Treiber | Ubuntu 24.04.3 / 580.95.05 | identisch |
| `/mnt/wbkolleg_dh_1` | gemountet | **gemountet, derselbe Share** — die 48 Job-Verzeichnisse von idhefix sind dort sichtbar |
| Systemplatte | 1,8 TB, 424 GB frei | 1,8 TB, 530 GB frei |
| `Linger` | `yes` | **`yes`** (16.09. gesetzt, ohne sudo) |
| HF-Cache | Symlink auf den Share | **Symlink gesetzt**, Schreibtest bestanden |
| sudo | kein passwortloses | kein passwortloses |

Das ändert die Ausgangslage: heute teilt sich ein Lauf **31,6 GB** mit den
Serving-Diensten; auf asteraix stehen **92 GB über NVLink** zur Verfügung.
Parallele Läufe und grössere Modelle sind damit keine Zukunftsmusik, sondern
der Grund für den Umzug — und bekommen einen eigenen Epic (T6), damit „umziehen"
und „ausbauen" sich nicht vermischen.

---

## 2. Entscheidungen (getroffen 16.09.2026)

- **E1 — der Share ist auf asteraix verfügbar.** Nachgewiesen, nicht nur zugesagt.
- **E2 — Rollen und Namen**: serving = idhefix, training = asteraix. Die
  Verwechslung hat sich durch 50 Dateien fortgepflanzt (§T0).
- **E3 — dünne Naht.** Kein geteiltes Vertragspaket. Der Proxy validiert nicht
  mehr selbst und leitet durch.
- **E4 — `eval/` zieht mit ins Trainingsrepo.**
- **E5 — die Registry wird geteilt, nicht über HTTP übergeben** (16.09., nach
  Rückfrage). Sie liegt auf dem Share, je trainiertem Modell eine Datei, atomar
  geschrieben. Das ersetzt `POST /admin/register` und löst zugleich die Frage,
  wie der Trainer einen `base_model` auflöst (§4 T3).

---

## 3. Was die Naht wirklich ist

Drei HTTP-Kanten, **keine** geteilte Python-Abhängigkeit — und seit E5
**kein** Netzaufruf für die Übergabe eines Modells. Gewichte und Registrierung
liegen beide auf dem Share:

| Kante | heute | nachher |
|---|---|---|
| `/train/*`-Proxy | Gateway → `127.0.0.1:8204` | idhefix → asteraix:8204 |
| Promotion-Gate | Trainer → `127.0.0.1:8200/ocr` | asteraix → idhefix:8200 |
| **`eval/`** | `eval/run_eval.py:44` → `127.0.0.1:8200/recognize` | asteraix → idhefix:8200 |

Und **eine vierte, die kein Import ist und den Split still überlebt**:
`manager._gpu_claim()` (`manager.py:399-412`) holt bei jedem vLLM-Start
`/gpu-claim` vom Trainer, mit 2 s Timeout, den `config.py:112` mit „the trainer
is on this box" begründet. Nach dem Split vergleicht dieser Wächter den Anspruch
einer **fremden** Maschine mit der **eigenen** Karte. Das ist kein Feature, das
fehlt — das ist ein Wächter, der falsche Auskunft gibt. Er gehört in T5 entfernt,
nicht in T2 umgebogen. **Entfernt am 16.09.2026** (T5.5, #139); ein Grep-Test
hält fest, dass kein Serving-Modul ihn wieder erwähnt.

Was **nicht** geteilt werden muss, entgegen der ersten Fassung:

- **`jobs_root` braucht keine Freigabe.** Der Gateway liest es nie: Logs
  (`train_routes.py:113`) und Kurven (`:123`) laufen über HTTP, und die
  Held-out-Seite des Gates quert als Multipart-Bytes (`promote.py:99-106`),
  nicht als Pfad.
- **`checkpoint_root` und der Arrow-Cache dürfen nicht auf den Share** — sie
  liegen lokal, *weil* der Share sie zerbrochen hat (`settings.py:43-50`:
  cross-device rename; `preflight.py:151-175`: 11½ Stunden verloren).

---

## 4. Epics

### T0 — Die Namensverwechslung auflösen

**Zuerst**, weil jedes später geschriebene Dokument den Fehler sonst weiterträgt.

Der Befund macht es einfacher als befürchtet: **keine** der 176 Fundstellen
meint die neue Box. Alle meinen 130.92.59.240. Ein stumpfes
`s/asterAIx/idhefix/` ist für **100 %** der heutigen Treffer sachlich richtig;
`asteraix` wird an Platzhaltern **eingefügt**, nirgends ersetzt.

| Issue | Inhalt |
|---|---|
| T0.1 | Ersetzung in beiden Repos, 50 Dateien. Einschliesslich `agentic_historian/bot.py:778` — das ist ein **im Discord sichtbarer** Slash-Command-Text, kein Kommentar |
| T0.2 | `docs/asteraix-environment.md` beschreibt die **Serving**-Box → `docs/idhefix-environment.md`. Der Name `asteraix` wird damit frei für die neue Box, die noch keine Dokumentation hat |
| T0.3 | Drei Sätze, die **eine Box für zwei Rollen** behaupten und aufgeteilt werden müssen, nicht ersetzt: `README.md:20`, `docs/EINFUEHRUNG.md:21`, `docs/TRAINING.md:1` |
| T0.4 | **Messwerte bleiben idhefix.** Jede Trainingszahl im Repo wurde dort gemessen — CER 0,466 Thun, 0,67 samples/s, die Vorfälle vom 14.09. Sie auf `asteraix` umzuschreiben würde den Messbericht fälschen |
| T0.5 | `~/.ssh/config`: `srv-train` → .240 ist doppelt irreführend. **Nicht blind umbenennen** — `Host ubelix` hängt mit `ProxyJump` daran (`ubelix/status.sh:9`, `docs/UBELIX_PLAN.md:35`). Alias `idhefix` ist angelegt; die Umstellung von `srv-train` braucht denselben Commit wie die UBELIX-Dateien |
| T0.6 | `docs/TRAINING_PLAN.md:71` behauptet „**No SSH from the dev machine**" — veraltet, es gibt einen funktionierenden Eintrag. Wer es glaubt, plant den Cutover als „kein Fernzugriff" |

**Tests**: `test_no_doc_calls_the_serving_box_asteraix` (Grep-Test über beide
Repos), `test_the_ubelix_proxyjump_alias_still_resolves`.

**Wichtig**: kein systemd-Unit, keine Env-Variable, kein Port und kein Hostname
enthält einen Boxnamen. T0 kann **nichts** kaputtmachen. Die einzige echte
Pfadbindung im ganzen Split ist `serving-atr-inference`, viermal fest in
`deploy/systemd/atr-train.service` (`:14,:15,:18,:23`) und einmal in
`.env.example:48` — eine **Repo**-Umbenennung, keine Box-Umbenennung.

### T1 — `training-atr-models` existiert und ist grün

| Issue | Inhalt |
|---|---|
| T1.1 | Repo, `pyproject.toml`, CI-Matrix wie hier, `.env.example` |
| T1.2 | `src/atr_serving/training/` → `src/atr_training/`, `engines/{kraken,trocr,vlm}_train_svc/` mit |
| T1.3 | Die **43** trainingsseitigen Tests plus `tests/conftest.py`. Achtung: conftest ist `autouse` für alle 72 Dateien, betrifft aber nur `TrainerSettings` — das Serving-Repo bleibt **ohne** conftest, und keiner der verbleibenden Tests liest sie |
| T1.4 | Die **einzige** Code-Kopplung auflösen: `atr_serving.registry` in `overlay.py:26`, `base_models.py:36` und vier Engine-Runnern. Alle vier Grenzgänger-Tests kreuzen über dasselbe Symbol |
| T1.5 | Neun Skripte mit Import + drei ohne (`make_split.py`, `smoke_trocr.py`, `check_env.sh`) |
| T1.6 | `ubelix/` — null Importarbeit, aber **14 Dateien** verdrahten `REPO=$HOME/serving-atr-inference` fest (`submit.sh:15` plus 13 sbatch), je mit `PYTHONPATH=$REPO/src:$REPO/engines` |
| T1.7 | `deploy/systemd/atr-train.service`, `.venvs/{kraken,trocr,vlm}-train`, `scripts/make_venvs.sh`. Dabei die Lücke schliessen: `scripts/check_venvs.sh:52-61` prüft **`trocr-train` nicht**, obwohl `make_venvs.sh:43` es baut |

**Tests**: die 43 bestehenden Dateien laufen unverändert durch.
`test_the_training_package_never_imports_atr_serving` — ein Import des anderen
Pakets lässt die Suite scheitern. Das ist die einzige Zusicherung, die
verhindert, dass die Naht über Monate wieder zuwächst.

### T2 — Der Proxy überquert das Netz

| Issue | Inhalt |
|---|---|
| T2.1 | `train_url` darf entfernt sein; Timeouts und ein ehrlicher 504 statt eines hängenden Requests |
| T2.2 | **Der Trainer hat heute gar keine Authentifizierung** — kein `Depends`, kein Key, keine Middleware; er verlässt sich vollständig auf den Loopback-Bind. Jede neue Kante quert zu einem ungeschützten Endpunkt. `X-API-Key` auf allen Routen + Bind ans VPN-Interface + `ufw`-Quellregel |
| T2.3 | Engine-Liste aus `/health`. Anmerkung: der Proxy importiert nicht `SUPPORTED_ENGINES`, sondern das rohe `BACKENDS` (`train_routes.py:26`) und baut das Tupel selbst (`:37`). Und `TrainerClient.health()` (`clients.py:302`) existiert, wurde aber **noch nie aufgerufen** — dies ist der Erstgebrauch einer toten Methode, mit 30 s Default-Timeout |
| T2.4 | `TrainRequest.model_validate` aus dem Proxy. **Zwei verschiedene Dinge wandern**: der 422 (`:90`, pydantic) und ein separater 400 (`:76`, Engine-Zugehörigkeit, feuert davor). Nur letzterer ändert den Status. Der Text des 400 ist zudem **falsch**: er behauptet „A TrOCR backend is planned … but not wired", obwohl TrOCR seit #44 registriert ist |
| T2.5 | Gemessen: der Trainer-422 ist `{"detail": [...]}`, und `clients.py:280` macht daraus per `str(detail)` einen Python-Repr, der pydantics vollständiges `input`-Echo mitschleppt — inklusive der injizierten kraken-VGSL-Vorgabe. Der Gateway streift das heute bei `:91` ab. Ohne Ersatz wird die Fehlermeldung **unlesbarer**, nicht nur anders |
| T2.6 | **`/train/gpu` scheitert nach dem Split still, nicht laut.** Die pids des entfernten Jobs (`:161`) werden gegen das **eigene** `/proc` gelaufen (`gpu.py:205-206`). Eine pid-Kollision markiert einen lokalen Fremdprozess als `registered` mit fremder job_id und nimmt ihn aus `unaccounted_mib` — genau der Fehlerfall, für den #414 existiert |
| T2.7 | Und es gibt **nichts, wohin man proxen könnte**: der Trainer hat kein `/gpu`. Seine `/health` liefert nur `GpuInfo(index, free_mb, total_mb)`. Die ganze Prozess-Zuordnung (`atr_serving/gpu.py`, 241 Zeilen nvidia-smi, `/proc` und systemd-Units) muss **dupliziert** werden — der Serving-Gateway braucht sie weiterhin für sein vLLM-Budget |

**Tests**: `test_a_remote_trainer_that_times_out_answers_504`,
`test_a_pid_on_the_training_box_is_never_attributed_to_a_local_process`,
`test_the_engine_list_comes_from_the_trainer_health`,
`test_a_validation_error_from_the_trainer_stays_readable`.

### T3 — Der Handover über die geteilte Registry

**Umgestellt am 16.09.2026.** Die erste Fassung wollte `POST /admin/register`.
Verworfen, weil die Einwände gegen eine geteilte Datei bei genauerer Prüfung
nicht halten — und die entscheidende Tatsache war schon im Code:

```
jobstore.save():  tmp.write_text(...)  →  os.replace(tmp, job_json)
jobs_root       = /mnt/wbkolleg_dh_1/…/jobs      ← derselbe Share
job.json        = 48 Stück, seit Wochen in Betrieb
```

**Atomares Ersetzen auf diesem CIFS-Mount ist im Betrieb bewiesen.**

| Einwand gegen eine geteilte Datei | Befund |
|---|---|
| Zwei Checkouts, zwei Dateien (`REPO_ROOT`-relativ) | Eine Env-Variable, die beide auf den Share zeigt |
| `save_overlay` schreibt nicht atomar (`overlay.py:82`) | Gilt für den heutigen Code, nicht für den Share — der Jobstore macht es auf demselben Mount richtig |
| Der Gateway liest nur beim Start (`app.py:42`) | Stimmt, **gilt für HTTP aber genauso** |
| Zwei Trainer parallel (T6) | Read-Modify-Write auf **eine** Datei verliert Updates → **eine Datei pro Modell**, wie der Jobstore eine pro Job hat |

```
/mnt/wbkolleg_dh_1/Textrecognition_Training/registry/
    models.yaml          ← kuratiert; Quelle bleibt Git im Serving-Repo;
                           vom Gateway beim Start veröffentlicht
    trained/<id>.yaml    ← vom Trainer geschrieben, tmp + os.replace
```

Was das gegenüber HTTP gewinnt: kein Endpunkt, keine Authentifizierung dafür,
**kein Fehlerfall „Gateway weg → Job scheitert nach 24 Stunden"** — und es
**löst T1.4 mit**: der Trainer liest `models.yaml` vom Share, ohne Kopie, die
auseinanderläuft, und sieht auch Modelle, die auf der Serving-Box deaktiviert
sind. `GET /models` hätte die versteckt (`routes.py:157` filtert auf `enabled`).

Mit E3 vereinbar: geteilt wird ein **Dateiformat**, kein Python-Paket. Ein
Vertragstest in beiden Repos pinnt das Schema.

Was es kostet:

- **Die Validierung wandert** vom Empfang zum Lesen. Eine kaputte Datei darf den
  Gateway nicht umwerfen — sie wird übersprungen und protokolliert. Die
  Aufteilung pro Modell macht das natürlich.
- **Nachladen ist unscharf.** CIFS-Clients cachen Attribute (`actimeo`); eine
  mtime-Prüfung kann um diese Frist nachhinken. Nach einem 24-Stunden-Lauf
  unerheblich.
- **Eine fehlende Registry muss laut scheitern.** Der Share ist sehr stabil —
  der Ausfall Ende August war eine angekündigte Wartung —, die Forderung hängt
  also nicht an der Häufigkeit, sondern an der Richtigkeit: ein Trainer, der eine
  Registry-ID auflösen muss und keine findet, darf nicht wie `load_heldout` still
  mit einer leeren weitermachen.

| Issue | Inhalt |
|---|---|
| serving#138 | Gateway veröffentlicht `models.yaml` (inklusive deaktivierter Einträge), liest `trained/` Datei für Datei, lädt ohne Neustart nach |
| training#14 | Die drei `_register` und `set_enabled` schreiben `trained/<id>.yaml` atomar; eine gescheiterte Registrierung lässt den Job scheitern und sagt, wo die Gewichte liegen |
| training#5 | `base_models.py` liest die Registry vom Share; die vorläufige `registry.py`-Kopie aus #3 verschwindet |

**Unverändert gültige Fallen:** vLLM bedient nie aus `local_path`
(`manager.py:81-88`), und ein nicht auflösbarer `local_path` meldet keine
fehlende Datei, sondern wird zu einer DOI-Suche (`kraken_loader.py:70-72`).
`merge_loras.py` bleibt auf der Serving-Seite; seine peft-Bindung an das venv,
„which trained it", spannt künftig über zwei Maschinen und wird von nichts
geprüft.

### T4 — `eval/` zieht mit

Der billigste Epic, und der Code stützt ihn: `eval/` hat **null** Serving-Importe.

| Issue | Inhalt |
|---|---|
| T4.1 | `eval/` und `textmetrics` ins Trainingsrepo. Danach hat `textmetrics` **keinen** Serving-Konsumenten mehr — die Frage nach einem geteilten Paket entfällt |
| T4.2 | `eval/` läuft heute mit dem **Gateway-venv** (`eval/README.md:10`); trainingsseitig braucht es `httpx` in einem eigenen venv |
| T4.3 | Was `eval/` von der Serving-Seite braucht, ist schmaler als gedacht: **eine Route** (`/recognize`) und **ein Schlüssel**. Nicht die Registry, nicht `models.yaml`, nicht `trained_root` — die Modell-ID ist ein opaker String (`run_eval.py:47`) |
| T4.4 | Zwei stehengebliebene Unwahrheiten in `eval/README.md`: die Begründung „#52 ist offen" (seit 08.08. beantwortet, ohne `eval/`) und „CER/WER liegen in `eval/metrics.py`" (seit `a8c8009` ein Re-Export) |
| T4.5 | `tests/test_eval.py` und `test_run_eval_corpus_cer.py` ziehen mit — in der ersten Fassung gar nicht aufgeführt |

### T5 — Cutover und Rückbau

**Cutover erledigt am 16.09.2026** (serving#139). idhefix trainiert nicht mehr.
Offen bleibt die Umstellung von asteraix auf den gemeinsamen Job-Speicher
(`jobs-asteraix` → `jobs`, nach training-atr-models#15) und der Abnahmetest:
v5 wird zum Schluss über den Gateway neu eingereicht (training-atr-models#10).

| Issue | Inhalt |
|---|---|
| T5.1 | Parallelbetrieb: `atr-train` auf asteraix hoch, Gateway zeigt noch auf Loopback |
| T5.2 | Smoke-Job (`trocr-thun-smoke`) über die neue Kante, alle fünf Stages inklusive Registrierung |
| T5.3 | `ATR_TRAIN_URL` umstellen, Gateway neu, `/train/jobs` **aus dem Bot heraus** prüfen. **Erledigt 16.09.2026, 19:22:** `ATR_TRAIN_URL` zeigt auf asteraix, Gateway mit #137 neu gestartet; `/train/jobs`, `/train/gpu` und `/health` über den Gateway geprüft, der ATR-MCP läuft ohne Konfigurationsänderung |
| T5.4 | Der alte Trainer auf idhefix wird gestoppt, nicht gelöscht. Die 48 Job-Records liegen ohnehin auf dem Share und sind von asteraix aus schon lesbar — **keine Migration nötig**. **Erledigt 16.09.2026, 21:3x:** `atr-train` gestoppt und deaktiviert (`disable --now`), vorher geprüft: 48 Jobs, keiner läuft, keiner wartet; auf `:8204` lauscht nichts mehr |
| T5.5 | **Rückbau der GPU-Koordination**: `/gpu-claim`, `/admin/release-gpu`, `gpu_release.py`, `manager._refuse_while_training`, `GpuBusyError`. Sie koordinieren zwei Prozesse um **eine** Karte. `manager._gpu_claim()` ist dabei der wichtigste Posten — er ist kein Import und würde den Split sonst still überleben. **Erledigt 16.09.2026** (#139), siehe unten |
| T5.6 | Der Artefakt-Cache **folgt den Daten nicht**. Derselbe Share heisst derselbe Ground Truth und derselbe HF-Cache, aber ein **kalter** `artefact_cache_root` — die ~2,5 h und ~41 GB je Auswahl, für die #109 existiert, werden auf der neuen Box einmal erneut bezahlt |

**T5.5 zuletzt.** Solange beide auf einer Karte laufen, ist die Sperre das, was
einen 24-Stunden-Lauf vor einer Inferenzanfrage schützt. Eingehalten: der
Rückbau kam nach T5.4.

**Was T5.5 am 16.09.2026 getan hat:**

- **Entfernt**, Gateway: der Aufruf des Trainer-Anspruchs vor jedem vLLM-Start
  (`manager._gpu_claim`), die Regel „Trainer antwortet nicht → nur auf eine
  leere Karte starten", `POST /admin/release-gpu` mit
  `ModelManager.release_lazy`, `gpu_claim_timeout_s`, und der Rückfall von
  `/train/gpu` auf eine lokale Messung. `/train/gpu` ist jetzt immer die
  Messung des Trainers; ein Trainer ohne `/gpu` ist ein 502 mit seiner URL.
- **Entfernt**, In-Repo-Trainer (deaktiviert, der Code bleibt bis zum Auszug des
  Pakets): `/gpu-claim` samt Cache, `training/gpu_release.py` und sein Aufruf an
  der Grenze zu `train` in `runner_base.py`.
- **Behalten:** die Platzprüfung vor jedem vLLM-Start — sie schützt vor den
  Engines und den Nachbarn —, die LRU-Verdrängung und das Autosizing.
  `GpuBusyError` bleibt als Ausnahme der Platzprüfung (503 mit `Retry-After`
  statt 502). Die Prüfung läuft jetzt **nach** der Verdrängung und verlangt,
  was auch das Autosizing verlangt (`vram_mb × 1,15` plus Reserve): vorher
  stand sie davor, und weil das Budget nie mehr als Karte minus Engines minus
  Reserve ist, lehnte sie jeden Start ab, der eine Verdrängung gebraucht hätte
  — auf idhefix war die LRU seit #129 toter Code.
- **Neu:** `GET /gpu` am Gateway — die Karten *dieser* Box, ohne
  Job-Zuordnung, mit denselben Zeilen wie das `/gpu` des Trainers, dazu `host`
  und `vllm` (die residenten Modelle mit `vram_mb`, die pids der eigenen
  vLLM-Kinder, das Budget). Seit #137 zeigte `/train/gpu` die Karten von
  asteraix, und wer dort die Last von idhefix abgelesen hatte, sah die
  Trainingsmaschine.
- **Budget:** `vllm_vram_budget_mb` ist kein fester Wert mehr, sondern wird bei
  jedem Start gelesen: Karte minus Engines (eigene Dienste, die nicht vom Gateway
  abstammen) minus Reserve. Gemessen 21:50 CEST: 46 068 − 15 830 − 2 048 =
  **28 190 MiB**; das alte 30 000 versprach 1 810 MiB, die die Karte nicht hat.
  Der Messwert ist der Rückfall ohne `nvidia-smi`, die Einstellung ein Override.

### T6 — Wofür der Umzug gemacht wird

Kein Migrationsschritt. Erst **nach** T5, damit der Umzug prüfbar bleibt.

| Issue | Inhalt |
|---|---|
| T6.1 | `max_concurrent: 1` → der Scheduler teilt **Karten** zu, statt eine vorauszusetzen. Heute pinnen alle fünf Units `CUDA_VISIBLE_DEVICES=1` und `params.device` ist `cuda:0` |
| T6.2 | VRAM-Vorprüfung **je Karte** statt global (`preflight.py`) |
| T6.3 | Ein Modell über beide Karten: NVLink mit 4 Links à 14,06 GB/s und P2P macht 92 GB nutzbar. Damit wäre Qwen3-VL-8B ohne 4-bit trainierbar — eine **Messung**, keine Annahme |
| T6.4 | Hyperparameter neu bestimmen. `batch_size: 1`, `accumulate_grad_batches: 16`, 4-bit QLoRA sind auf 31,6 GB geteilte Karte abgestimmt |
| T6.5 | Sechs Kommentare in `src/atr_serving/training/` begründen Verhalten mit **idhefix'** Hardware (`preflight.py:4-9`, `settings.py:119`, `hf_source.py:8`, `contracts.py:305`, `runner_base.py:1113`). Nach dem Umzug beschreiben sie eine Maschine, auf der der Code nicht mehr läuft — insbesondere `preflight.py`s Begründung „GPU 1 is shared with the serving engines", die der Split gerade aufhebt |

---

## 5. Teststrategie

Drei Testarten, die es heute nicht gibt:

1. **Isolationstest je Repo** — ein Import des anderen Pakets lässt die Suite
   scheitern. Die einzige Zusicherung gegen erneutes Zuwachsen.
2. **Kontrakttests an der Naht** — beide Repos halten dieselbe Sammlung von
   `/train/*`-Anfragen und -Antworten vor. Ohne sie ist E3 ein Versprechen ohne
   Prüfung. Besonders: `TrainEngine` (`contracts.py:103`) und die Engine-Liste
   des Gateways sind heute **von Hand** einig, ohne Test.
3. **Netzwerkfehler-Tests je Route** — Trainer weg, langsam, 500. Heute
   unnötig, danach der häufigste Fall.

---

## 6. Reihenfolge

```
T0 (Namen)  →  T1 (Repo grün)  →  T2 (Proxy)  ┐
                                 T3 (Handover) ┼→  T5 (Cutover)  →  T5.5 Rückbau  →  T6 (Ausbau)
                                 T4 (eval/)    ┘
```

---

## 7. Offen

- **Zwei Hosts im selben HF-Cache.** Heute schreiben bereits zwei *Projekte*
  hinein (dieses und `lassberg/vlm_training`); zwei *Maschinen* ist ungeprüft.
  Das dokumentierte CIFS-Problem ist Dedup, ausdrücklich „harmless but not
  optimal" — nicht Nebenläufigkeit. Der Code hält weder Lock-Disziplin noch eine
  Beobachtung dazu fest: **unbeantwortet, nicht schlecht beantwortet.**
- ~~**Ob ein laufender Job den Cutover übersteht.**~~ **Entschieden am
  16.09.2026, 21:50 CEST** (#139): Ein laufender Job wird bei einer Umstellung
  **nicht migriert** — er bleibt auf der Maschine, die ihn gestartet hat;
  Checkpoints und TMPDIR liegen lokal. Bei **unter etwa 6 h Restlaufzeit wartet
  die Umstellung**: der alte Trainer bleibt bis zum Ende aktiv, `ATR_TRAIN_URL`
  wird erst danach umgestellt. **Sonst wird der Lauf mit Ansage abgebrochen** und
  auf der neuen Maschine neu eingereicht — billig nur mit warmem Artefakt-Cache
  (T5.6), sonst wieder rund 1 h (v5 auf asteraix: prepare 13:59–15:04, compile
  bis 15:59). Die Entscheidung steht **mit Job-ID und Restlaufzeit im
  Umstellungs-Issue**.
- **UBELIX**: zieht mit T1.6 um, aber ob asteraix es ersetzt oder ergänzt, ist
  offen.
