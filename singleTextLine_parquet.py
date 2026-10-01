from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET
import os
import re


# ==============================
# KONFIGURATION
# ==============================

PARQUET_PATH = Path(
    
)

COLUMN_NAME = "xml_content"
SUFFIX = "_singleTextLine"

# Womit die ursprünglichen Zeilen im Fliesstext verbunden werden.
# " "  -> ein durchgehender Fliesstext ohne Zeilenumbrüche (Reintext-Test)
# "\n" -> Zeilenumbrüche bleiben innerhalb der einen TextLine erhalten
JOIN_WITH = " "

# Namespace für das NEU geschriebene PAGE-XML (derselbe wie in der Pipeline)
PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"

print("Parquet-Pfad:", PARQUET_PATH)


# ==============================
# XML → PLAIN TEXT
# ==============================

def local_name(tag):
    return tag.rsplit("}", 1)[-1]


def extract_lines(root):
    """Text jeder <TextLine>, in Dokumentreihenfolge.

    Pro TextLine zählt nur ihr eigenes <TextEquiv>/<Unicode> – nicht die der
    <Word>- oder <Glyph>-Elemente darin
    """
    lines = []
    for line in root.iter():
        if local_name(line.tag) != "TextLine":
            continue
        for child in line:
            if local_name(child.tag) != "TextEquiv":
                continue
            for unicode_el in child:
                if local_name(unicode_el.tag) == "Unicode" and unicode_el.text and unicode_el.text.strip():
                    lines.append(unicode_el.text.strip())
                    break
            break
    return lines


# ==============================
# PLAIN TEXT → PAGE-XML MIT EINER <TextLine>
# ==============================

def build_single_textline(text, image_filename="", width="0", height="0"):
    """Minimales PAGE-XML: ein TextRegion, eine TextLine, darin der ganze Text.

    Keine Coords/Baseline – die Pipeline braucht sie für VLM auf Seitenebene
    nicht.
    """
    ET.register_namespace("", PAGE_NS)
    pcgts = ET.Element(f"{{{PAGE_NS}}}PcGts")
    page = ET.SubElement(pcgts, f"{{{PAGE_NS}}}Page", {
        "imageFilename": image_filename or "",
        "imageWidth": str(width or 0),
        "imageHeight": str(height or 0),
    })
    region = ET.SubElement(page, f"{{{PAGE_NS}}}TextRegion", {"id": "r1"})
    line = ET.SubElement(region, f"{{{PAGE_NS}}}TextLine", {"id": "l1"})
    equiv = ET.SubElement(line, f"{{{PAGE_NS}}}TextEquiv")
    ET.SubElement(equiv, f"{{{PAGE_NS}}}Unicode").text = text
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(pcgts, encoding="unicode")


def to_single_textline(xml_content):
    if pd.isna(xml_content) or not str(xml_content).strip():
        return xml_content

    image_filename, width, height = "", "0", "0"
    try:
        root = ET.fromstring(xml_content)
        lines = extract_lines(root)
        page = next((e for e in root.iter() if local_name(e.tag) == "Page"), None)
        if page is not None:
            image_filename = page.get("imageFilename", "")
            width = page.get("imageWidth", "0")
            height = page.get("imageHeight", "0")
        if not lines:
            print("WARNUNG: Keine transkribierte <TextLine> gefunden.")
            return ""
    except ET.ParseError:
        # Kein XML -> ist schon reiner Text (z.B. eine bereits bereinigte Datei)
        lines = [str(xml_content).strip()]

    if JOIN_WITH == " ":
        text = re.sub(r"\s+", " ", " ".join(lines)).strip()
    else:
        text = JOIN_WITH.join(lines)

    return build_single_textline(text, image_filename, width, height)


# ==============================
# PARQUET EINLESEN
# ==============================

if __name__ == "__main__":
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"Datei nicht gefunden: {PARQUET_PATH}"
        )

    df = pd.read_parquet(PARQUET_PATH)

    if COLUMN_NAME not in df.columns:
        raise KeyError(
            f"Spalte '{COLUMN_NAME}' nicht gefunden. "
            f"Vorhandene Spalten: {list(df.columns)}"
        )

    print(f"Datei: {PARQUET_PATH.name}")
    print(f"Anzahl Zeilen: {len(df)}")
    print(f"Bearbeite Spalte: {COLUMN_NAME}")

    # ==============================
    # IN EINE <TextLine> UMWANDELN
    # ==============================

    df[COLUMN_NAME] = df[COLUMN_NAME].apply(to_single_textline)

    empty = int((df[COLUMN_NAME].fillna("") == "").sum())
    if empty:
        print(f"WARNUNG: {empty} Zeilen ohne Text – die Pipeline überspringt diese Seiten.")

    # ==============================
    # MIT NEUEM TITEL SPEICHERN
    # ==============================

    output_path = PARQUET_PATH.with_name(
        f"{PARQUET_PATH.stem}{SUFFIX}{PARQUET_PATH.suffix}"
    )

    temp_path = output_path.with_name(
        output_path.stem + ".tmp" + output_path.suffix
    )

    df.to_parquet(temp_path, index=False)

    os.replace(temp_path, output_path)

    print(f"Fertig. Parquet gespeichert unter:")
    print(output_path)
