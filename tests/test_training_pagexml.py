"""PageXML prepare-stage helpers (#33)."""

from pathlib import Path

import pytest

from atr_serving.training.pagexml import (
    PageXMLError,
    has_transcription,
    image_filename,
    line_texts,
    page_stats,
    rewrite_image_filename,
)

FIXTURE = Path(__file__).parent / "fixtures" / "page_sample.xml"


@pytest.fixture
def xml() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def test_reads_image_filename(xml):
    assert image_filename(xml) == "023499_0012_623887.jpg"


def test_rewrite_round_trips(xml):
    out = rewrite_image_filename(xml, "000001_023499_0012_623887.jpg")
    assert image_filename(out) == "000001_023499_0012_623887.jpg"
    # the rest of the document is untouched — line count and content survive
    assert line_texts(out) == line_texts(xml)


def test_rewrite_preserves_everything_else(xml):
    out = rewrite_image_filename(xml, "p.jpg")
    assert out.startswith('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>')
    assert 'xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"' in out
    assert 'imageWidth="1600" imageHeight="1067"' in out
    # only the one attribute changed
    assert out.replace("p.jpg", "023499_0012_623887.jpg") == xml


def test_rewrite_rejects_a_path(xml):
    with pytest.raises(PageXMLError):
        rewrite_image_filename(xml, "sub/dir/p.jpg")


def test_rewrite_escapes_the_new_name(xml):
    out = rewrite_image_filename(xml, 'a"b&c.jpg')
    assert 'imageFilename="a&quot;b&amp;c.jpg"' in out
    assert image_filename(out) == "a&quot;b&amp;c.jpg"  # raw attribute text


def test_rewrite_without_a_page_element():
    with pytest.raises(PageXMLError):
        rewrite_image_filename("<PcGts></PcGts>", "p.jpg")


def test_line_texts_ignores_page_level_text(xml):
    texts = line_texts(xml)
    assert len(texts) == 3
    assert texts[0] == "Item ontfaen van Janne"
    assert texts[1] == "van der Straten & sinen sone"  # entity decoded by the parser
    assert texts[2] == ""
    assert "page level text that is not a line" not in texts


def test_page_stats_counts_only_transcribed_lines(xml):
    stats = page_stats(xml)
    assert stats.lines == 3
    assert stats.transcribed_lines == 2
    assert stats.chars == sum(len(t) for t in line_texts(xml) if t.strip())
    assert {"I", "&", "J"} <= stats.charset
    assert stats.usable


def test_page_without_transcription_is_droppable():
    xml = (
        '<PcGts><Page imageFilename="a.jpg">'
        '<TextLine><TextEquiv><Unicode>  </Unicode></TextEquiv></TextLine>'
        '<TextLine/>'
        "</Page></PcGts>"
    )
    stats = page_stats(xml)
    assert stats.lines == 2 and stats.transcribed_lines == 0
    assert not stats.usable
    assert not has_transcription(xml)


def test_namespaced_prefix_form_is_supported():
    xml = (
        '<pc:PcGts xmlns:pc="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">'
        '<pc:Page imageFilename=\'x.jpg\'>'
        "<pc:TextLine><pc:TextEquiv><pc:Unicode>hallo</pc:Unicode></pc:TextEquiv></pc:TextLine>"
        "</pc:Page></pc:PcGts>"
    )
    assert image_filename(xml) == "x.jpg"
    assert line_texts(xml) == ["hallo"]
    assert image_filename(rewrite_image_filename(xml, "y.jpg")) == "y.jpg"


def test_unparsable_xml_raises():
    with pytest.raises(PageXMLError):
        line_texts("<Page><TextLine>")
