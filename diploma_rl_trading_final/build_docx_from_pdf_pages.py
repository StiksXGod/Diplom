from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent
PAGES_DIR = ROOT / "docx_pages"
OUTPUT_PATH = ROOT / "diploma_rl_trading.docx"

# A4 in twips and EMU.
PAGE_WIDTH_TWIPS = 11906
PAGE_HEIGHT_TWIPS = 16838
TWIP_TO_EMU = 635
PAGE_WIDTH_EMU = PAGE_WIDTH_TWIPS * TWIP_TO_EMU
PAGE_HEIGHT_EMU = PAGE_HEIGHT_TWIPS * TWIP_TO_EMU


def build_paragraph_with_image(rel_id: str, image_name: str, docpr_id: int) -> str:
    return f"""
    <w:p>
      <w:pPr>
        <w:jc w:val="center"/>
        <w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/>
      </w:pPr>
      <w:r>
        <w:drawing>
          <wp:inline distT="0" distB="0" distL="0" distR="0">
            <wp:extent cx="{PAGE_WIDTH_EMU}" cy="{PAGE_HEIGHT_EMU}"/>
            <wp:effectExtent l="0" t="0" r="0" b="0"/>
            <wp:docPr id="{docpr_id}" name="Page {docpr_id}"/>
            <wp:cNvGraphicFramePr>
              <a:graphicFrameLocks noChangeAspect="1"/>
            </wp:cNvGraphicFramePr>
            <a:graphic>
              <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
                <pic:pic>
                  <pic:nvPicPr>
                    <pic:cNvPr id="0" name="{image_name}"/>
                    <pic:cNvPicPr/>
                  </pic:nvPicPr>
                  <pic:blipFill>
                    <a:blip r:embed="{rel_id}"/>
                    <a:stretch>
                      <a:fillRect/>
                    </a:stretch>
                  </pic:blipFill>
                  <pic:spPr>
                    <a:xfrm>
                      <a:off x="0" y="0"/>
                      <a:ext cx="{PAGE_WIDTH_EMU}" cy="{PAGE_HEIGHT_EMU}"/>
                    </a:xfrm>
                    <a:prstGeom prst="rect">
                      <a:avLst/>
                    </a:prstGeom>
                  </pic:spPr>
                </pic:pic>
              </a:graphicData>
            </a:graphic>
          </wp:inline>
        </w:drawing>
      </w:r>
    </w:p>
    """.strip()


def build_page_break() -> str:
    return """
    <w:p>
      <w:pPr>
        <w:spacing w:before="0" w:after="0"/>
      </w:pPr>
      <w:r>
        <w:br w:type="page"/>
      </w:r>
    </w:p>
    """.strip()


def build_document_xml(images: list[Path]) -> str:
    blocks: list[str] = []
    for index, image_path in enumerate(images, start=1):
        blocks.append(
            build_paragraph_with_image(
                rel_id=f"rId{index}",
                image_name=image_path.name,
                docpr_id=index,
            )
        )
        if index != len(images):
            blocks.append(build_page_break())

    body = "\n".join(blocks)
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document
  xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas"
  xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"
  xmlns:o="urn:schemas-microsoft-com:office:office"
  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
  xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"
  xmlns:v="urn:schemas-microsoft-com:vml"
  xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing"
  xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
  xmlns:w10="urn:schemas-microsoft-com:office:word"
  xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
  xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"
  xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup"
  xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk"
  xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml"
  xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
  xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
  xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"
  mc:Ignorable="w14 wp14">
  <w:body>
    {body}
    <w:sectPr>
      <w:pgSz w:w="{PAGE_WIDTH_TWIPS}" w:h="{PAGE_HEIGHT_TWIPS}"/>
      <w:pgMar w:top="0" w:right="0" w:bottom="0" w:left="0" w:header="0" w:footer="0" w:gutter="0"/>
    </w:sectPr>
  </w:body>
</w:document>
"""


def build_relationships(images: list[Path]) -> str:
    items = [
        f'<Relationship Id="rId{index}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
        f'Target="media/{image_path.name}"/>'
        for index, image_path in enumerate(images, start=1)
    ]
    joined = "\n  ".join(items)
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {joined}
</Relationships>
"""


def build_content_types() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="jpeg" ContentType="image/jpeg"/>
  <Default Extension="jpg" ContentType="image/jpeg"/>
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""


def build_root_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""


def build_core_props() -> str:
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>diploma_rl_trading</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:modified>
</cp:coreProperties>
"""


def build_app_props(page_count: int) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
  <Pages>{page_count}</Pages>
  <Words>0</Words>
  <Characters>0</Characters>
</Properties>
"""


def main() -> None:
    images = sorted(PAGES_DIR.glob("page-*.jpg"))
    if not images:
        raise FileNotFoundError(f"No page images found in {PAGES_DIR}")

    document_xml = build_document_xml(images)
    relationships_xml = build_relationships(images)

    with ZipFile(OUTPUT_PATH, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", build_content_types())
        archive.writestr("_rels/.rels", build_root_rels())
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/_rels/document.xml.rels", relationships_xml)
        archive.writestr("docProps/core.xml", build_core_props())
        archive.writestr("docProps/app.xml", build_app_props(len(images)))
        for image_path in images:
            archive.write(image_path, f"word/media/{image_path.name}")

    print(f"Created {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
