# Einführung: Handschriftenerkennung trainieren

Für Studierende, die neu in diesem Projekt sind. Diese Seite erklärt, **was hier
passiert und warum**. Wenn du bereits weisst, was CER, PageXML und QLoRA sind und
nur einen Lauf starten willst, brauchst du stattdessen
[`TRAINING.md`](TRAINING.md) oder [`VLM_TRAINING.md`](VLM_TRAINING.md).

---

## 1. Das Problem

Historische Handschriften sind maschinell schwer lesbar. Ein Modell, das
neuzeitliche Druckschrift fehlerfrei erkennt, scheitert an einer Berner Missive
von 1550 — andere Buchstabenformen, Abkürzungen, verblasste Tinte, unregelmässige
Zeilen.

Die Aufgabe heisst **ATR** (*Automatic Text Recognition*) oder **HTR**
(*Handwritten Text Recognition*). Eingabe: das Bild einer Seite. Ausgabe: der Text
darauf.

Dieses Repository betreibt dafür einen Server auf der Maschine **asterAIx** und
kann Modelle nicht nur *benutzen*, sondern auch **selbst trainieren**.

## 2. Zwei Wege, und warum es beide gibt

Es gibt zwei grundverschiedene Ansätze, und dieses Projekt verfolgt beide.

### kraken (CTC)

Ein vergleichsweise kleines neuronales Netz (~15 Millionen Parameter), das eine
Zeile von links nach rechts abtastet und pro Position ein Zeichen vorhersagt. Die
Technik heisst **CTC** (*Connectionist Temporal Classification*) und löst das
Problem, dass man vorher nicht weiss, wo im Bild welcher Buchstabe endet.

Es bildet **Bildinformation auf Zeichen ab** — mehr nicht. Es weiss nichts über
Sprache, kann also auch nichts erfinden.

### VLM (Vision-Language-Model)

Ein grosses vortrainiertes Modell (hier Qwen3-VL mit 8 Milliarden Parametern),
das Bilder *und* Text versteht. Man zeigt ihm eine Zeile und die Anweisung
„transkribiere das", und es antwortet in Worten.

Weil es Sprache kennt, produziert es plausibles Deutsch — **auch dort, wo es das
Bild nicht lesen kann.** Das ist Stärke und Schwäche zugleich.

### Was die Messungen zeigen

Beide auf derselben Evaluationsmenge, 189 Zeilen aus Thun:

| Modell | Verfahren | Trainingszeilen | CER |
|---|---|---:|---:|
| `thun-kurrent-v2` | kraken | 1 898 | **0,218** |
| `qwen3vl-german-medieval-v1` | VLM | 4 124 | 0,232 |

Das VLM hatte **doppelt so viele Daten und war trotzdem schlechter.** Interessanter
als die Gesamtzahl ist aber, *wie* die Fehler sich verteilen: das VLM lässt weniger
aus, fügt dafür mehr hinzu und verwechselt häufiger Zeichen. Es liest mehr von der
Zeile — und mehr davon falsch. Genau das erwartet man von einem Sprachmodell, das
plausibel rät, wo ein CTC-Netz schweigen würde.

## 3. Wie Trainingsdaten aussehen

Das Rohmaterial sind **Seitenbilder plus PageXML** — ein XML-Format, das für jede
Textzeile speichert, *wo* sie auf der Seite liegt und *was* dort steht:

```xml
<TextLine id="l17">
  <Coords points="112,884 1043,884 1043,952 112,952"/>
  <TextEquiv><Unicode>Item ontfaen van Janne</Unicode></TextEquiv>
</TextLine>
```

Die `Coords` umschreiben ein Polygon, der `Unicode` ist die Transkription, die ein
Mensch angefertigt hat. Solche Daten entstehen in **Transkribus** und liegen bei
uns auf **HuggingFace** — 32 Datensätze der Gruppe `dh-unibe`, von mittelalterlichen
Urkunden bis zu Bundesratsprotokollen des 19. Jahrhunderts.

Aus einer Seite werden viele **Zeilen**, und die Zeile ist die Trainingseinheit.
Ein Korpus von 12 000 Seiten ergibt rund 325 000 Zeilen.

## 4. Was ein Trainingslauf tut

Fünf Stufen, immer dieselben, für beide Verfahren:

| Stufe | was passiert |
|---|---|
| **prepare** | Seiten von HuggingFace holen und auf die Platte schreiben |
| **compile** | daraus das Format machen, das der Trainer liest (Zeilenausschnitte) |
| **train** | das eigentliche Training |
| **test** | das fertige Modell gegen zurückgehaltene Daten messen |
| **register** | das Modell ablegen, damit der Server es ausliefern kann |

Man reicht einen Job als JSON über eine HTTP-Schnittstelle ein und fragt seinen
Zustand ab. **Es läuft immer nur ein Job**, weil sich eine GPU nicht gut teilen
lässt.

## 5. Die Kennzahl: CER

**CER** = *Character Error Rate*, der Anteil falscher Zeichen.

- CER 0,00 — fehlerfrei
- CER 0,10 — jedes zehnte Zeichen falsch; für viele Zwecke brauchbar
- CER 0,22 — unser bester deutscher Wert; lesbar, aber mühsam
- CER 0,98 — das Modell hat nichts gelernt

> ### ⚠️ Eine Falle, die alle einmal treffen
>
> In diesem Projekt bedeuten die Fehlerarten **das Gegenteil des Üblichen**:
>
> - `insertions` = Zeichen, die **fehlen**
> - `deletions` = Zeichen, die **hinzugefügt** wurden
>
> Das kommt von kraken und ist in `tests/test_edit_convention.py` festgeschrieben —
> **nicht „korrigieren"**. Die Doku dieses Projekts hat die Konvention selbst
> monatelang falsch gelesen und deshalb einen Fehlschlag genau falsch herum
> erklärt.

## 6. Was wir gelernt haben

Vier Befunde aus echten Läufen. Sie sind nicht offensichtlich, und drei davon
widersprechen der ursprünglichen Erwartung.

### Die Schriftklasse schlägt das Jahrhundert

Für Thun (16. Jh.) war ein **Kurrent**-Basismodell aus dem 16./17. Jahrhundert
40 % besser als ein **Textura**-Modell derselben Epoche. Textura ist eine
Buchschrift, Kurrent eine Gebrauchsschrift — und unser Material ist
Gebrauchsschrift. *Wie* geschrieben wurde zählt mehr als *wann*.

### Fast nie von Null anfangen

Der erste Lauf trainierte ein Netz von zufälligen Gewichten aus auf 1 898 Zeilen.
Ergebnis: CER 0,98, praktisch leere Ausgabe. Unter etwa 100 000 Zeilen nimmt man
ein vorhandenes Modell und passt es an (*Fine-Tuning*) — nicht, weil es bequemer
wäre, sondern weil es sonst nicht funktioniert.

### Irgendwann sind mehr Epochen wertlos

Von 30 auf 90 Epochen brachte 7 % Verbesserung bei dreifachem Rechenaufwand. Wenn
die Kurve flach wird, hilft kein längeres Training — dann fehlen **Daten**.

### Der Datensatz, nach dem man greift, ist oft der falsche

Der naheliegende Datensatz — `image-text_medieval-scripts_xiv-xv-xvi`, 548 000
Seiten — ist laut seiner eigenen Beschreibung **flämisch**. Sein deutscher Anteil
sind 291 brauchbare Seiten. Die 9 885 Seiten Zürcher Richtebücher lagen die ganze
Zeit in einem anderen Datensatz, den niemand abgefragt hatte.

Deshalb gibt es `scripts/plan_corpus.py`: es bewertet alle 32 Datensätze nach
Epoche, Sprache und Schriftklasse und entfernt Projekte, die zwei Datensätze
doppelt veröffentlichen — mehrere tun das.

## 7. Dein erster Lauf

**Fang klein an.** Ein Lauf über den vollen Korpus dauert Tage; einer über
40 Seiten dauert Minuten und lehrt dasselbe über den Ablauf.

```bash
curl -s -X POST "http://localhost:8200/train/jobs" \
  -H "X-API-Key: $(grep ^ATR_API_KEY .env | cut -d= -f2)" \
  -H "Content-Type: application/json" -d '{
    "model_id": "mein-erster-versuch",
    "engine": "kraken",
    "base_model": "kraken-early_modern_german",
    "dataset": {
      "hf_repo": "dh-unibe/image-text_medieval-scripts_xiv-xv-xvi",
      "train_projects": ["GT_Thun-Training_(TEST-DEMO)"],
      "eval_projects": ["GT_Thun-Test_(DEMO_TEST)"],
      "max_pages": 40
    },
    "params": {"batch_size": 16, "resize": "union", "epochs": 20}
  }' | python3 -m json.tool
```

Dann zusehen:

```bash
curl -s localhost:8204/jobs/<job-id> | python3 -m json.tool | head -30
```

**`verify_only=true` an die URL hängen prüft alles, ohne etwas zu starten.**
Gewöhn dir das an — es kostet Sekunden und erspart Stunden.

## 8. Wenn etwas schiefgeht

Das ist der Normalfall, nicht die Ausnahme. Vier von fünf Läufen in der
Entwicklungsphase sind gescheitert, und jeder Fehlschlag hat eine Schutzmassnahme
hervorgebracht.

Die Fehlermeldung steht immer im Job-Record unter `error`, und sie ist
absichtlich ausführlich:

```bash
curl -s localhost:8204/jobs/<job-id> | python3 -c 'import json,sys; print(json.load(sys.stdin)["error"])'
```

Häufige Ursachen:

- **`CUDA out of memory`** — die Batchgrösse passt nicht auf die Karte, die wir
  uns mit anderen Diensten teilen. Halbieren hilft nicht immer: kraken füllt jeden
  Batch auf seine *breiteste* Zeile auf, eine einzige fehlsegmentierte Zeile kann
  also mehr kosten als alle anderen zusammen.
- **`429 Too Many Requests`** — HuggingFace begrenzt auf 1 000 Anfragen je fünf
  Minuten, und ein Korpus mit vielen Projektverzeichnissen reisst das.
- **Der Lauf refüsiert vor dem Start** — mit Begründung. Es gibt Schutzmassnahmen
  gegen zu wenige Optimizer-Schritte, zu wenig Plattenplatz, zu wenig VRAM und
  ungültige Basismodelle. Sie sind Freunde: sie kosten Sekunden, wo ein Fehlschlag
  Stunden kostet.

## 9. Weiterlesen

| Datei | wofür |
|---|---|
| [`TRAINING.md`](TRAINING.md) | Anleitung für kraken-Läufe, Überwachung, Fehlersuche |
| [`VLM_TRAINING.md`](VLM_TRAINING.md) | dasselbe für VLM, mit Laufzeitrechnung |
| [`TRAINING_PLAN.md`](TRAINING_PLAN.md) §9 | alle Messungen mit Begründung — hier steht das *Warum* |
| [`../README.md`](../README.md) | Überblick über den ganzen Server |

Und wenn du eine Zahl berichtest: **sag immer dazu, gegen welche
Evaluationsmenge sie gemessen wurde.** Eine CER von 0,22 auf Thun und eine von
0,22 auf Königsfelden sind nicht dieselbe Aussage, und die Verwechslung ist der
häufigste Fehler beim Vergleichen von Modellen.
