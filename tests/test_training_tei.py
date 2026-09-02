"""Reading a TEI edition as page-level training material (#91).

The St. Gallen missives are an edition, not a PageXML corpus: no coordinates, so
no line crops. Page-level training does not need them — `vlm_dataset.page_sample`
reads only `line_texts` — but the conversion has to separate what is on the page
from what an editor wrote about it.
"""

import pytest

from atr_serving.training.pagexml import line_texts
from atr_serving.training.tei import (
    SKIP_SUBTREES,
    TeiError,
    TeiPage,
    page_texts,
    to_pagexml,
)

NS = 'xmlns="http://www.tei-c.org/ns/1.0"'


def tei(body: str, header: str = "<persName>Stefan Sonderegger</persName>") -> str:
    return (f"<TEI {NS}><teiHeader>{header}</teiHeader>"
            f"<text><body><p>{body}</p></body></text></TEI>")


class TestPageAndLineStructure:
    def test_pb_starts_a_page_and_names_its_image(self):
        pages = page_texts(tei('<pb facs="A.JPG" n="1"/><lb n="1"/>eins'
                               '<pb facs="B.JPG" n="2"/><lb n="1"/>zwei'))
        assert [p.image for p in pages] == ["A.JPG", "B.JPG"]
        assert pages[0].lines == ("eins",) and pages[1].lines == ("zwei",)

    def test_lines_keep_the_manuscripts_breaks(self):
        """A diplomatic transcription's line breaks are on the page, so they stay."""
        p = page_texts(tei('<pb facs="A.JPG"/><lb n="1"/>erste<lb n="2"/>zweite'))[0]
        assert p.lines == ("erste", "zweite")
        assert p.text == "erste\nzweite"

    def test_a_word_split_across_lines_stays_split(self):
        """`<lb break="no"/>` means the word continues — and it does, on the page."""
        p = page_texts(tei('<pb facs="A.JPG"/><lb n="1"/>burger<lb break="no" n="2"/>maister'))[0]
        assert p.lines == ("burger", "maister")

    def test_pretty_printed_whitespace_is_collapsed(self):
        """TEI files are indented; the indentation is not part of the text."""
        p = page_texts(tei('<pb facs="A.JPG"/><lb n="1"/>Min   undertaenig\n'
                           '                willig dienst'))[0]
        assert p.lines == ("Min undertaenig willig dienst",)

    def test_a_page_without_facs_is_dropped(self):
        """Without the image name there is nothing to pair the text with."""
        pages = page_texts(tei('<pb n="1"/><lb/>orphan<pb facs="B.JPG"/><lb/>kept'))
        assert [p.image for p in pages] == ["B.JPG"]


class TestOnThePageVersusAboutIt:
    def test_ner_content_is_kept_and_its_tags_dropped(self):
        """persName wraps text that is written on the page."""
        p = page_texts(tei('<pb facs="A.JPG"/><lb/>an <persName ref="x">Cuni am Wasen</persName> gericht'))[0]
        assert p.lines == ("an Cuni am Wasen gericht",)

    def test_a_line_break_inside_a_name_still_breaks_the_line(self):
        """`<orgName>burgermaister und <lb/>rat ze sant Gallen</orgName>` is real."""
        p = page_texts(tei('<pb facs="A.JPG"/><lb n="1"/>'
                           '<orgName>burgermaister und <lb n="2"/>rat ze sant Gallen</orgName>'))[0]
        assert p.lines == ("burgermaister und", "rat ze sant Gallen")

    def test_editorial_notes_are_excluded(self):
        """"Es ist unklar, welche Person gemeint ist" is about the page, not on it.

        Keeping it would train the model to invent footnotes.
        """
        p = page_texts(tei('<pb facs="A.JPG"/><lb/>der herr'
                           '<note>Es ist unklar, welche Person gemeint ist.</note>'))[0]
        assert p.lines == ("der herr",)

    def test_the_text_after_a_note_is_not_lost_with_it(self):
        """A note interrupts a sentence that continues; the tail is page text."""
        p = page_texts(tei('<pb facs="A.JPG"/><lb/>es wollen<note>Unklar.</note> heute'))[0]
        assert p.lines == ("es wollen heute",)

    def test_the_header_contributes_nothing(self):
        """persName in revisionDesc is the transcriber, not a person on the page."""
        p = page_texts(tei('<pb facs="A.JPG"/><lb/>seitentext'))[0]
        assert "Sonderegger" not in p.text

    def test_every_skipped_subtree_behaves_the_same_way(self):
        for tag in sorted(SKIP_SUBTREES - {"teiHeader"}):
            p = page_texts(tei(f'<pb facs="A.JPG"/><lb/>vor<{tag}>WEG</{tag}>nach'))[0]
            assert "WEG" not in p.text, tag
            assert p.lines == ("vornach",), tag


class TestToPageXml:
    def test_the_text_survives_the_roundtrip(self):
        page = TeiPage("A.JPG", ("erste zeile", "zweite zeile"))
        assert line_texts(to_pagexml(page)) == ["erste zeile", "zweite zeile"]

    def test_it_names_the_image_the_edition_named(self):
        assert 'imageFilename="StadtASG_Missive_1_1.JPG"' in to_pagexml(
            TeiPage("StadtASG_Missive_1_1.JPG", ("x",)))

    def test_markup_characters_in_the_transcription_are_escaped(self):
        """Editions contain < and & often enough that this is not theoretical."""
        page = TeiPage("A.JPG", ('a < b & "c"',))
        assert line_texts(to_pagexml(page)) == ['a < b & "c"']

    def test_it_carries_no_coordinates(self):
        """Line granularity over this yields nothing, which is correct: say
        `granularity: "page"` and mean it, rather than silently falling back."""
        assert "Coords" not in to_pagexml(TeiPage("A.JPG", ("x",)))

    def test_an_empty_page_is_refused_rather_than_written(self):
        with pytest.raises(TeiError, match="no transcribed line"):
            to_pagexml(TeiPage("A.JPG", ()))


class TestRefusals:
    def test_a_non_tei_document_says_so(self):
        with pytest.raises(TeiError, match="no <body>"):
            page_texts('<PcGts><Page/></PcGts>')

    def test_malformed_xml_says_so(self):
        with pytest.raises(TeiError, match="not parseable"):
            page_texts("<TEI><body>")
