# CHURRO für deutsche Handschriften des 14.–16. Jahrhunderts — Trainingsplan

Stand: 11.09.2026. Ausgangspunkt ist die Literaturübersicht *„Feinabstimmung von
Vision-Language-Modellen für historische Schriften"* (Google Doc, 2026). Alles,
was dieser Plan über CHURRO behauptet, ist an den Primärquellen nachgeprüft —
Model Card, Datensatz, Quellcode — und an unserem eigenen Korpus nachgemessen.
Wo die Übersicht und die Primärquellen auseinandergehen, steht es in §1.

---

## 0. Was „ein CHURRO-Modell trainieren" hier heissen kann

[`stanford-oval/churro-3B`](https://hf.co/stanford-oval/churro-3B) ist ein
Qwen2.5-VL-3B (3,75 Mrd. Parameter), vollständig feinabgestimmt auf
[CHURRO-DS](https://hf.co/datasets/stanford-oval/churro-dataset): 97.200
Trainings-, 1.200 Dev- und 1.200 Testseiten aus 155 Sammlungen.

Drei Lesarten, und nur eine ist verhältnismässig:

| | Was | Aufwand | Urteil |
|---|---|---|---|
| a | CHURRO nachbauen: Full-Parameter-Training auf CHURRO-DS | ~6.000 H100-Stunden (Übersicht); auf UBELIX 4× H100 rund zwei Monate, 134,5 GB Trainingsdaten | **nein** — die Gewichte sind offen, Nachbauen erzeugt nichts Neues |
| b | **CHURRO als Basis nehmen und an unser Material anpassen** | eine LoRA auf ~14.000 Seiten, Tage statt Monate | **ja** |
| c | CHURROs Rezept auf unseren Daten, ab Qwen2.5-VL-3B | wie (b) | **als Kontrollarm** — ohne ihn ist ein gutes Ergebnis von (b) nicht CHURRO zuzuschreiben |

(b) ist genau die Arbeitsteilung, mit der die Übersicht selbst schliesst: grosse
Konsortien erzeugen historische Basismodelle, kleinere Archive passen sie per PEFT
an lokale Bestände an.

---

## 1. Was die Übersicht nicht sagt, aber alles verändert

Die Übersicht stützt sich für CHURRO auf eine Sekundärquelle (einen „Literature
Review" auf themoonlight.io). Vier Punkte fehlen dort oder stehen anders:

### 1.1 CHURRO spricht XML, nicht Klartext

Aus `src/churro_ocr/templates/presets.py` im
[CHURRO-Repo](https://github.com/stanford-oval/Churro):

```python
CHURRO_3B_XML_TEMPLATE = HFChatTemplate(
    system_message="Transcribe the entirety of this historical document to XML format.",
    user_prompt=None,
)
```

Ausgabe ist das eigene `HistoricalDocument`-Format — Metadaten, dann `<Page>` mit
`<Header>`, `<Body>`, `<Footer>` und `<Line>`-Elementen, dazu Inline-Markup für
Hinzufügungen, Tilgungen und Lücken. `DEFAULT_OCR_MAX_TOKENS` steht dort auf
**25.000**.

Und das Trainingsziel war genau das: die Spalte `cleaned_transcription` in
CHURRO-DS beginnt so —

```xml
<HistoricalDocument xmlns="http://example.com/historicaldocument">
  <Metadata>
    <Language>Czech</Language>
    <WritingDirection>ltr</WritingDirection>
    <PhysicalDescription>Printed book page from the 18th or 19th century, Švabach script.</PhysicalDescription>
```

**Folgen:** (1) Eine Auswertung gegen unsere Ground Truth braucht einen
XML→Text-Schritt, sonst zählt jedes Tag als Fehler. (2) Unser Trainer lernt heute
Klartext mit dem Prompt „Transcribe the handwritten text in this image exactly as
written." — CHURRO damit weiterzutrainieren hiesse, gegen sein eigenes Vorwissen
anzulernen. (3) CHURRO wurde darauf trainiert, eine **frei formulierte
`PhysicalDescription` zu erzeugen** — Text, der auf der Seite nicht steht.

### 1.2 Die Lizenz ist nicht-kommerziell

CHURRO erbt von Qwen2.5-VL-3B die **Qwen Research License**: Nutzung,
Weitergabe und Ableitungen „FOR NON-COMMERCIAL PURPOSES ONLY". Weitergabe
verlangt die Lizenz, eine `NOTICE`-Datei („Qwen is licensed under the Qwen
RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All Rights Reserved.")
und Änderungshinweise.

Zum Vergleich: **Qwen3-VL-8B (Basis von v3) und Qwen3-VL-4B stehen unter Apache
2.0**, ebenso Qwen2.5-VL-7B. Der 3B ist in der Familie der einzige mit dieser
Einschränkung. Ein Wechsel auf CHURRO führt sie ein; für Forschung an der UniBE
ist das zulässig, `publish.py` muss sie aber mitliefern (es erfindet bewusst
keine Lizenz).

### 1.3 „Normalisierung" ist so nicht belegt

Die Übersicht schreibt, CHURRO habe „störende typografische Eigenheiten
normalisiert". CHURROs eigener Standard-Prompt verlangt das Gegenteil: *„Do not
modernize or standardize the text. For example, if the transcription is using ‚ſ'
instead of ‚s' … keep it that way."* Das „cleaned" in `cleaned_transcription`
ist, soweit sichtbar, die Umformung in das XML-Schema. Ob dabei auch Orthografie
angeglichen wurde, ist an Beispielzeilen zu prüfen (§3, Schritt 0.1) — nicht
anzunehmen.

### 1.4 Deutsch ist dabei, aber nicht unser Deutsch

Im Testsplit sind **60 von 1.200 Seiten** deutsch (5 %; der Split ist erkennbar
auf 60 Seiten je Hauptsprache geschichtet). Die deutschen Quellen dort:

- Handschrift: `stabs`, `kurfurst`, `klostern`, `faithful`, `konzils`, `icdar2017`
- Druck: `ocrd`, `nzz`, `germany`, `reichsanzeiger`, `impact`, `fondue-art`

Keine davon trägt einen Namen, der zu unseren fünf Quellen passt (Zürcher
Rats- und Richtebücher, Bullinger, Königsfelden, AAEB, St. Galler Missiven).
Den 97.200-seitigen Trainingssplit habe ich **nicht** durchgezählt — die
Kontaminationsfrage ist damit wahrscheinlich, aber nicht belegt verneint (§3,
Schritt 0.2).

---

## 2. Das eigentliche Risiko: unsere Transkriptionskonvention

Gemessen an den 13.929 Seiten und 12,3 Mio. Zeichen des v3-Korpus:

| Zeichen | Anzahl | je 1.000 Zeichen | Lesart |
|---|---:|---:|---|
| `✳` | **70.376** | 5,72 | projektspezifische Markierung |
| `ˀ` | 23.506 | 1,91 | projektspezifisch (Kürzungszeichen?) |
| `ù` | 23.408 | 1,90 | vermutlich Notation, kein Gravis |
| `₎` | 5.905 | 0,48 | projektspezifisch |
| kombinierendes Makron `̄` | 46.798 | 3,81 | Kürzungsstrich |
| kombinierendes `ͤ` / `ͦ` | 18.404 / 19.900 | 1,50 / 1,62 | übergeschriebene Vokale |
| alle kombinierenden Zeichen | 93.249 | 7,58 | |
| `ſ` (langes s) | **547** | 0,04 | fast immer zu `s` normalisiert |

### Woher die Zeichen kommen (Schritt 0.1, gemessen am 11.09.)

Je Quelle, zugeordnet über die Transkribus-`docId` im Seitennamen — ein Dokument
gehört nie zu zwei Datasets. Gezählt sind nur Seiten, deren Zuordnung eindeutig
ist (11.332 von 13.929, siehe Anmerkung unten):

| Quelle | `✳` | `ˀ` | `ù` | `₎` | `ſ` | Makron | `ͤ` | `ͦ` | `✳` je 1.000 Zeichen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Königsfelden** (charters post 1500) | 30.603 | 11.605 | 11.183 | 3.659 | 0 | 29.638 | 11.758 | 8.961 | **14,40** |
| Zürcher Rats- und Richtebücher | 3.491 | 3.700 | 1.952 | 248 | **547** | 1.633 | 1.399 | 2.216 | 1,85 |
| Bullinger | 0 | 0 | 0 | 0 | 0 | 0 | 790 | 1.530 | 0 |
| AAEB | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| St. Galler Missiven | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**Die Notation ist im Kern die des Königsfelden-Korpus**: es trägt 76–95 % jedes
Sonderzeichens (`✳` 90 %, `₎` 94 %, Makron 95 %) in acht- bis zehnfacher Dichte
der nächsten Quelle. **Aber nicht ausschliesslich**: die Zürcher Rats- und
Richtebücher verwenden dieselben Zeichen sparsamer und sind die einzige Quelle
mit `ſ`. Bullinger kennt nur übergeschriebene Vokale, AAEB und die Missiven sind
vollständig normalisiert.

Das heisst für jedes Modell, nicht nur für CHURRO: der Korpus mischt mindestens
drei Konventionen — diplomatisch mit Sonderzeichen (Königsfelden, Rats- und
Richtebücher), übergeschriebene Vokale (Bullinger), normalisiert (AAEB, Missiven).
Ein Modell lernt, je nach Handschrift eine andere Notation zu schreiben.

**Entscheidung (11.09.):** vorerst alles beibehalten — auch in CHURROs Training.

> Anmerkung zur Messung: Der Pool-Index vorn im Seitennamen ist **kein
> eindeutiger Schlüssel** über Datasets hinweg. `_prepare_multi` setzt
> `start_index=total_pages_written`, zählt also nur geschriebene Seiten, während
> übersprungene Seiten innerhalb eines Datasets trotzdem Indizes verbrauchen. Das
> nächste Dataset beginnt daher im Indexbereich des vorigen. Die Dateinamen
> bleiben dank `docId`/`pageId` eindeutig, aber für 2.597 Seiten (19 %) lässt
> sich die Quelle aus dem Index allein nicht bestimmen. Ein erster, naiver
> Durchgang schrieb AAEB deshalb 7.411 `✳` zu, die in Wahrheit Königsfelden-Seiten
> waren.

Zwei Konsequenzen:

**Ein Zero-Shot-Ergebnis von CHURRO misst zuerst Konvention, nicht Lesefähigkeit.**
CHURRO hat `✳`, `ˀ` oder `₎` nie gesehen und schreibt `ſ`, wo unsere Ground
Truth `s` hat. Allein `✳` sind 0,57 % aller Zeichen — ein CER-Sockel, den kein
Lesen beseitigt. Die Zahl ist daher **zweimal** zu erheben: roh, und nach einer
Abbildung beider Seiten auf eine gemeinsame Konvention.

**Die Hauptarbeit der Feinabstimmung ist womöglich Konvention, nicht Schrift.**
Das wäre gut: Notation lernt ein Modell schnell. Aber ob alle fünf Quellen
dieselbe Konvention teilen, ist unbekannt — wenn `✳` aus einer einzigen Quelle
stammt, trainieren wir heute schon (auch in v3) eine Mischung, die jedes Modell
bestraft. Das betrifft nicht nur CHURRO.

Ob `✳` & Co. behalten oder vereinheitlicht werden, ist eine **editorische**
Entscheidung — `✳` kann eine sinntragende Markierung sein —, keine technische.
Entschieden ist sie vorerst für **Beibehalten** (§5).

---

## 3. Der Plan

### Phase 0 — messen, bevor irgendetwas trainiert wird (≈ ½ Tag, kein Training)

| Schritt | Was | Warum |
|---|---|---|
| 0.1 | Konventionsprofil **je Quelle**: welche Quelle trägt `✳`, `ˀ`, `₎`, `ſ`, Makron; dazu 20 CHURRO-DS-Beispielzeilen `original_` vs. `cleaned_transcription` | klärt §1.3 und §2 |
| 0.2 | `dataset_id`-Liste des CHURRO-Trainingssplits gegen unsere fünf Quellen | Kontamination (§1.4) |
| 0.3 | `HistoricalDocument`-XML → Klartext, als getestete Funktion: `<Line>` in Lesereihenfolge mit Zeilenumbruch, `Metadata` verwerfen, Inline-Markup nach einer festen Regel auflösen | ohne das ist jede CHURRO-Zahl falsch |
| 0.4 | **Zero-Shot** auf den 1.391 v3-Validierungsseiten, zuerst 200: `churro-3B` mit seinem eigenen Prompt, und `Qwen2.5-VL-3B-Instruct` als Nullpunkt. CER roh und konventionsbereinigt, `length_ratio`, NLS | die Zahl, an der sich alles entscheidet |

Für 0.4 nicht die #92-Lehre vergessen: `max_new_tokens` mindestens 4.096, und
`_looks_truncated` einschalten. Die halbierte CER von `qwen3vl-sg-missiven-v1`
(0,5921 → 0,2785) war abgeschnittene Ausgabe, kein schlechtes Modell.

**Gate G0** nach Phase 0:

- CHURRO zero-shot, bereinigt, ≲ v3 → die LoRA muss vor allem Konvention lehren; kleiner Aufwand, grosse Erwartung.
- deutlich schlechter als v3 → CHURROs Vorwissen trägt auf unsere Kanzleischriften nicht; Phase 1–3 nur als Experiment, nicht als Ersatz für v3.
- zero-shot bereits sehr gut, aber mit Halluzinationen (`length_ratio` > 1) → das Metadaten-/Beschreibungsverhalten aus §1.1 ist der Hauptgegner.

### Phase 1 — Pipeline anpassen (≈ 1 Tag)

| Schritt | Was |
|---|---|
| 1.1 | **Zielformat XML statt Klartext.** Aus den `line_texts`, die `page_sample` schon heute liest, ein `HistoricalDocument` mit `<Page><Body><Line>…` bauen; CHURROs Systemprompt übernehmen. So bleibt die Feinabstimmung auf CHURROs Vorwissen, statt es umzuschreiben. Klartext + unser Prompt nur als Ablation. |
| 1.2 | **Metadaten minimal**: `Language`, `WritingDirection` — **keine `PhysicalDescription`**. Wir wollen dem Modell nicht beibringen, Text über die Seite zu erfinden, der nicht auf ihr steht. |
| 1.3 | **Visuelles Budget neu kalibrieren.** Qwen2.5-VL rechnet in 28-px-Zellen (Patch 14 × Merge 2), Qwen3-VL in 32-px. `apply_visual_budget` leitet die Zelle schon aus dem Prozessor ab, aber `VLM_PIXEL_BUDGET["page"] = 2048·32²` Pixel ergibt auf 28-px-Zellen **~2.675** statt 2.048 Token. Laut UBELIX (Exp. A/C) ist genau das der Durchsatzhebel. |
| 1.4 | **bf16 statt 4-bit.** Ein 3B braucht in bf16 ~7,5 GB Gewichte und passt neben die Serving-Engines auf die A40; UBELIX Exp. B: 4-bit kostet 23 % Durchsatz ohne Nutzen. |
| 1.5 | Generierungsbudget für XML ≥ 4.096; `VLM_MAX_SAMPLE_CHARS` (#110) bleibt, XML-Overhead einrechnen. |
| 1.6 | Smoke-Lauf auf dem Thun-Set, wie bei jedem neuen Backend. |

`base_model` ist schon heute ein freies Feld, und `AutoModelForImageTextToText`
lädt `qwen2_5_vl` — CHURRO als Basis erfordert keinen Umbau des Laders, nur 1.1–1.5.

### Phase 2 — LoRA-Feinabstimmung auf asterAIx

Daten wie v3: dieselbe Auswahl (5 Datasets, 13.929 Seiten nach #110), **derselbe
Split** (seed 42, partition 0,9), damit jede Zahl direkt neben v3 steht.

| Parameter | Wert | Begründung |
|---|---|---|
| Basis | `stanford-oval/churro-3B` | |
| LoRA | r 64, α 128, alle 7 Projektionen | wie v3 — ein Unterschied weniger, der das Ergebnis erklärt |
| Lernrate | **1·10⁻⁴**, cosine, 5 % Warm-up | zwischen unseren 2·10⁻⁴ und Ketabas 2·10⁻⁵: CHURRO ist schon domänenangepasst, zu hohe Raten riskieren, sein Vorwissen zu überschreiben. Erster Kandidat für einen Sweep, falls `eval_loss` unruhig ist |
| Batch | 1 × 16 | wie v3 |
| Epochen | 1, `max_epochs` 3, `patience` 1 | wie v3; Continuation (#88) entscheidet |
| Präzision | bf16 | §1.4 |
| Recovery | automatisch alle 50 Schritte (#119) | |

**Zeitschätzung**, aus eigenen Messungen statt aus der Übersicht: v3 braucht 48 s
je Schritt à 16 Seiten, also ~3 s/Seite (8B, 4-bit, A40). UBELIX Exp. C: 4B ist
nur 1,5 % schneller als 8B — die Zeit steckt im Vision-Tower, nicht in der
Sprachmodell-Grösse. Mit bf16 (−23 %) ergibt das **~2,3 s/Seite, ~8 h je Epoche**,
1–3 Epochen also 8–25 h. **Ein 3B wird hier kaum schneller sein als v3.** Sein Wert
ist das Vorwissen, nicht die Geschwindigkeit.

Zusätzlich **Arm D** mit identischem Rezept ab `Qwen2.5-VL-3B-Instruct`. Wegen
`max_concurrent: 1` laufen C und D nacheinander.

### Phase 3 — Vergleich und Entscheidung

Alle Arme auf denselben 1.391 Validierungsseiten, dieselbe Auswertung:

| Arm | Basis | Training | Frage |
|---|---|---|---|
| A | Qwen3-VL-8B | QLoRA (= v3) | der heutige Stand |
| B | CHURRO-3B | keins | wie viel bringt CHURRO ohne Arbeit? |
| C | CHURRO-3B | LoRA, Phase 2 | der Kandidat |
| D | Qwen2.5-VL-3B | LoRA, gleiches Rezept | **was davon ist CHURRO und was bloss „irgendein 3B"?** |

Gemessen wird mit `textmetrics` — **mit unserer invertierten Konvention**
(`insertions` = in der Hypothese fehlend, `deletions` = hinzugefügt) —, dazu
`length_ratio` als Halluzinations- und Normalisierungsanzeiger und NLS für den
Vergleich mit den 70,1 % des Papers.

- **C < A deutlich** → CHURRO wird Basis für deutsche Seiten.
- **C ≈ D** → CHURROs historisches Vortraining trägt für unser Material nichts bei; dann entscheidet die Lizenz (§1.2) für Qwen3-VL.
- **A am besten** → bei v3 bleiben; UBELIX Exp. C hatte 2B schon 31 % schlechter als 4B/8B gezeigt, ein 3B liegt dazwischen.

### Phase 4 (nur bei Bedarf) — Full-Parameter auf UBELIX

Nur, wenn C klar über A plateaut **und** Grund besteht, die LoRA-Kapazität für die
Grenze zu halten. 3,75 Mrd. Parameter mit Adam in bf16 sind ~60 GB plus
Aktivierungen: FSDP über die 4× H100 (preemptable, kostenlos) oder 8-bit-Adam auf
einer 94-GB-Karte. **Voraussetzung:** Wiederaufnahme aus einem Snapshot. #119
schreibt heute Snapshots, aber nichts setzt von ihnen fort — auf einer Partition,
die Jobs abbricht, ist das kein Detail.

---

## 4. Was ich aus der Übersicht bewusst **nicht** übernehme

| Vorschlag | Warum nicht (jetzt) |
|---|---|
| LLM- oder ByT5-Nachkorrektur | Die Übersicht selbst belegt das „Komplexitäts-Paradoxon": jede zusätzliche Komponente verschlechterte das Ergebnis. |
| Ensembles (Linear+Boost, Top-5-Voting) | `agentic_historian` #416 hat gemessen, dass Fusion das beste Einzelmodell nicht schlägt, wo eines klar führt. |
| DoRA / RSLoRA (Ketaba) | Erst eine Basislinie, dann Neuerungen — nicht mehrere Variablen auf einmal verändern. Kandidat für eine Ablation nach Phase 3. |
| Elastische Verformung, Scherung | Die CER-Gewinne stammen aus zeilenbasiertem TrOCR (Gwalther). Bei seitenbasierten VLMs mit 14.000 Seiten nicht der erste Hebel. |
| Zahlen aus der Übersicht als Erwartung | 70,1 % NLS gelten für CHURRO-DS über 46 Sprachen, nicht für Schweizer Kanzleischrift des 15. Jh. Unsere Erwartung ist die v3-CER, sonst nichts. |

---

## 5. Entscheidungen (getroffen am 11.09.2026)

| # | Frage | Entscheidung | Folge |
|---|---|---|---|
| 1 | Wann Phase 0.4 | **auf GPU 1, sobald sie frei ist** — nach v3 und dem TrOCR-Smoke-Job | GPU 0 bleibt dem RAG-Dienst der Nachbarn; ein Warteskript auf der Box startet 0.4 von selbst, wenn die Trainingsqueue leer ist |
| 2 | Zielformat | **XML** (`HistoricalDocument`) | Phase 1.1 wie beschrieben; Klartext nur als Ablation |
| 3 | Konvention | **vorerst alles beibehalten** | kein Mapping der Ground Truth; die Zero-Shot-Zahl wird trotzdem zusätzlich konventionsbereinigt erhoben, um Lesen von Notation zu trennen. Herkunft der Zeichen: §2 |
| 4 | Lizenz | **„nur nicht-kommerziell" genügt** — Einsatz ist universitäre Forschung, Lehre und nicht-kommerzielle Anwendung, etwa mit Archiven | `publish.py` liefert `qwen-research`-`LICENSE` und `NOTICE` mit; ein kommerzieller Einsatz bräuchte eine Lizenz von Alibaba Cloud |

## 6. Zeitrahmen

| | |
|---|---|
| heute Nacht, ~01:30 CEST | v3-Test-CER = Arm A |
| 12.09. | Phase 0 (½ Tag) → Gate G0 |
| 13.09. | Phase 1 (1 Tag) |
| 13.–16.09. | Phase 2: Arme C und D nacheinander, je 8–25 h |
| danach | Phase 3 |

Rund eine Woche Wanduhrzeit, fast ganz von der GPU bestimmt. Phase 4 ist nicht
eingeplant.

---

**Quellen:** [churro-3B Model Card](https://hf.co/stanford-oval/churro-3B) ·
[CHURRO-DS](https://hf.co/datasets/stanford-oval/churro-dataset) ·
[CHURRO-Repo](https://github.com/stanford-oval/Churro) (`templates/presets.py`,
`prompts/ocr.py`, `providers/specs.py`) · [arXiv 2509.19768](https://arxiv.org/abs/2509.19768) ·
Qwen2.5-VL-3B `LICENSE` · eigene Messungen am v3-Korpus
(`20260910T110352Z-qwen3vl-german-pages-v3`) · `docs/UBELIX_PLAN.md` Exp. A–C.
