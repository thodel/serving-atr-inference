from pathlib import Path
import xml.etree.ElementTree as ET

# ==============================
# KONFIGURATION
# ==============================

BASE_DIR = Path.cwd()

XML_PATH = BASE_DIR / "input"
TXT_OUTPUT_DIR = BASE_DIR / "output"

TXT_OUTPUT_DIR.mkdir(exist_ok=True)

print("XML-Pfad:", XML_PATH)
print("TXT-Ausgabeordner:", TXT_OUTPUT_DIR)

NAMESPACES = {"tei": "http://www.tei-c.org/ns/1.0"}



for xml_file in XML_PATH.glob("*.xml"):
    print(f"Verarbeite: {xml_file.name}")
    tree = ET.parse(xml_file)
    root = tree.getroot()

    #Textbody finden

    p = root.find(".//tei:text/tei:body/tei:p", NAMESPACES)

    if p is None:
        print(f"Kein Textbody in {xml_file.name} gefunden.")
        continue

    plain_text = "".join(p.itertext())

    plain_text = " ".join(plain_text.split())

    output_file = TXT_OUTPUT_DIR / f"{xml_file.stem}.txt"
    output_file.write_text(plain_text, encoding="utf-8")

    print("Text extrahiert und gespeichert in:", output_file)
    