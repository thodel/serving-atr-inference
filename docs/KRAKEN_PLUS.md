# kraken+ — Ströbels HTR+-Nachbau, und was davon für uns gilt

Quelle: Ströbel, *diss_stroebel_v2.pdf*, Kap. 3.6.1 (S. 93 der PDF / Druckseite 71),
Ergebnisse in Tab. 3.1 (S. 94), Tab. 3.2 (S. 96), Tab. 3.6 (S. 100).

## Die Definition

> „Additionally, we rebuilt the HTR+ model from Figure 3.8 in VGSL in kraken (we call
> the rebuild kraken+)“

```
kraken+ = [256,64,0,1 Cr4,2,8,4,2 Cr4,2,32,1,1 Mp4,2,4,2 Cr3,3,64,1,1
           Mp1,2,1,2 S1(1x0)1,3 Lbx256 Do0.5 Lbx256 Do0.5 Lbx256 Do0.5
           Cr255,1,85,1,1]
```

Begleitende Hyperparameter aus demselben Abschnitt: **batch size 256**, **lrate 0.0001**,
**cyclical learning rate** (Smith 2017), Adam, Eingangshöhe 64 px, Dropout 0.5 nach jeder
BLSTM-Schicht. Konvergenz „after 18 epochs“ ohne weitere Verbesserung auf dem
Validierungsset.

Das ist exakt die Spezifikation, mit der dieses Projekt gestartet ist.

## Was die 85 bedeuten — und warum sie nicht übertragbar sind

Ströbel begründet die letzte Schicht ausdrücklich:

> „The number of filters in the last convolutional layer of the HTR+ depends on the number
> of different characters. Since we cannot control for this number in VGSL […] we examined
> the frequency distribution of the characters in the training set. We determined that
> limiting the number of filters to 85 (out of 144 different characters […]) is a
> reasonable threshold.“

**85 ist also ein korpusspezifischer Wert** — 85 von 144 Zeichen eines Frakturkorpus, nach
Häufigkeit abgeschnitten. Für unser Material ist er nicht einfach zu übernehmen: der
Codec des mediävistischen Korpus umfasst **102 Klassen** (101 Zeichen + Blank). Eine
85-Filter-Schicht davor ist ein Rang-Engpass — die Logits von 102 Klassen werden durch
einen 85-dimensionalen ReLU-Raum gezwängt, und kraken hängt die eigentliche
Ausgabeschicht ohnehin automatisch an („CTC layer will be added automatically“).

## Was `Cr255,1,85,1,1` in kraken 7.0.2 tatsächlich tut

`Cr<y>,<x>,<d>,<sy>,<sx>` heißt Kernelhöhe 255, Kernelbreite 1, 85 Filter. Nach
`S1(1x0)1,3` ist die Feature-Map genau **eine Zeile hoch**. `ActConv2D` paddet mit
`(kernel-1)//2` (`kraken/lib/vgsl/layers.py:805`), hier also 127 Zeilen oben und unten:

* Ausgabehöhe bleibt 1,
* **nur die mittlere Kernelzeile sieht Daten**, die übrigen 254 sehen ausschließlich
  Nullpadding und bekommen nie einen Gradienten,
* gemessen: 11.097.600 von 15.193.853 Parametern (255 × 512 × 85) liegen in dieser
  Schicht, **effektiv wirksam sind davon 43.520** (512 × 85).

Die Schicht wirkt also wie eine 1×1-Faltung mit 11 Mio. toten Gewichten. Das ist
Verschwendung, aber kein Fehler im Sinne von „funktioniert nicht“ — Ströbels eigene
Ergebnisse belegen, dass das Modell trainiert.

## Ströbels Ergebnisse — und warum sie kein Zielwert für uns sind

| Modell | F1 (bag-of-words) | CER % |
|---|---|---|
| HTR+ | 0.970 | 0.67 |
| **kraken+** | **0.954** | **0.89** |
| PyLaia | 0.969 | 0.60 |
| TrOCR | 0.990 | 0.89 |
| kraken (default) | 0.865 | — |

kraken+ liegt 1.6 Prozentpunkte hinter HTR+; Ströbel führt das auf **Bildvorverarbeitung
und Datenaugmentierung** zurück (Deslope, Deslant, Binarisierung, Dilatation, Distortion),
die in HTR+/PyLaia immer laufen.

**Entscheidend: das ist gedruckte Fraktur (NZZ), nicht Handschrift.** CER 0.89 % auf
Zeitungsdruck sagt nichts über handgeschriebene mediävistische Vorlagen. Die
Ablation (Tab. 3.6) zeigt zudem, wie stark die Werte an der Materialgüte hängen: 150
Seiten → 0.89 %, 12 Seiten → 3.61 %, 200 Zeilen → 5.04 %.

## Vergleich mit unseren Läufen

| Lauf | Spezifikation | lrate | Ergebnis |
|---|---|---|---|
| run 1 | **kraken+ wortgetreu** | 1e-4, 1cycle | val_accuracy **0.000**, 11 Epochen Blank-Collapse |
| run 2 | kraken+ **ohne** `Cr255,1,85,1,1` | 1e-3, 1cycle | val 0.7809 (Ep. 80), Test-CER 0.181 |
| run 3 | kraken-Default (120 px, `Lbx200`×3) | 1e-3, 1cycle | val **0.8226** (Ep. 60) |

**Die Architektur ist damit nicht widerlegt.** Der einzige wortgetreue kraken+-Lauf
scheiterte an der Lernrate, nicht am Netz: unter `1cycle` mit `div_factor=25` startet
lrate 1e-4 bei 4e-6 und lag nach 11 Epochen erst bei ~3.3e-5. Zwischen run 1 und run 2
wurden **zwei** Dinge geändert (Lernrate *und* Schlussschicht), also ist der Beitrag der
Schicht bislang unbekannt.

Bemerkenswert: Ströbel fährt mit demselben lrate 1e-4 und zyklischer Lernrate erfolgreich
und konvergiert in 18 Epochen. Unterschiede, die das erklären können: gedruckte statt
handschriftlicher Vorlagen, andere kraken-Version (die 1cycle-Parametrisierung ist nicht
dieselbe) und ein deutlich einfacheres Zeichenrepertoire.

## Umsetzbar? Ja. Sinnvoll? Gemessen — und die Antwort ist differenzierter als erwartet.

Gemessen am 2026-09-01, alle drei Läufe auf `shard_00` trainiert, gegen `val_clean`
validiert und gegen dasselbe Held-out-Set getestet (6.186 Seiten, 35 ungesehene
Dokumente):

| Lauf | Spezifikation | Höhe | val acc | **Test-CER** | Word acc |
|---|---|---:|---:|---:|---:|
| run 2 | ohne Schlussschicht | 64 | 0.7809 | 0.1812 | 51.8 % |
| **kraken+** | `Cr1,1,85,1,1` | 64 | **0.7927** | **0.1655** | 55.0 % |
| run 3 | kraken-Default | 120 | **0.8226** | **0.1335** | 62.5 % |

**Die 85-Kanal-Schlussschicht schadet nicht — sie hilft leicht.** kraken+ liegt auf
beiden Metriken über run 2, das dieselbe Architektur ohne diese Schicht hat. Ein
Zwischenstand bei Epoche 25 sah umgekehrt aus (kraken+ 0.6593 gegen run 2s 0.6765);
die Kurven haben sich später gekreuzt, und eine Frühphasen-Momentaufnahme trägt diese
Frage nicht.

Der Rang-Engpass, den wir befürchtet hatten — 85 ReLU-Kanäle vor 102 Klassen —
kostet also nichts Messbares. Ströbels korpusspezifische 85 sind für unser Material
kein Problem, obwohl sie für ein anderes Zeichenrepertoire gewählt wurden.

**Was tatsächlich trennt, ist die Eingangshöhe.** Die beiden 64-px-Läufe liegen 0.0157
CER auseinander, der 120-px-Lauf liegt 0.032 unter dem besseren von ihnen. Das deckt
sich mit der Geometrie (S10): bei stride 8 hat 64 px nur 1.97 CTC-Frames pro Zeichen,
120 px hat 3.69.

Die Fehlerformen dazu:

| | Insertions | Deletions | Substitutions |
|---|---:|---:|---:|
| run 2 | 433.428 | 366.997 | 545.953 |
| kraken+ | 333.579 | 382.316 | 514.324 |
| run 3 | 279.767 | 348.331 | 364.368 |

Alle drei sind konvergierte Modelle (Substitutionen dominieren), und alle drei
erkennen **kein einziges** der 170 `Inherited`-Zeichen — der kombinierenden
Kürzungszeichen. Das ist kein Architekturproblem dieser Größenordnung.

### Was das für die Praxis heißt

kraken+ nachzubauen war die Mühe wert, aber nicht als Kandidat für ein Produktivmodell:
der kraken-Default ist auf diesem Material deutlich besser, bei praktisch gleicher
Parameterzahl und einem Viertel der FLOPs. Der Wert des Nachbaus liegt darin, zwei
Hypothesen erledigt zu haben — die tote Kernelzeile (11,1 Mio. Parameter, die nur
Nullpadding sehen) und den 85-Kanal-Engpass —, die beide als Erklärung für run 1s
Scheitern im Raum standen. Keine von beiden war es; es war die Lernrate (#96).

**Alle Zahlen hier sind auf flämischem Material gemessen** (`medieval-scripts` sind
Leuvener Register). Die Rangfolge der Architekturen gilt, die absoluten CERs sind
keine Aussage über deutsche Handschrift — dafür #98.
