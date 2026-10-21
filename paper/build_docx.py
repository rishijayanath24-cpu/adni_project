"""
Build the TCSST/IRO Word manuscript from the LaTeX source, on top of the
journal's own Word template (styles, numbering, page setup are the template's).

    python paper/build_docx.py  [template.docx]  [out.docx]

Equations are converted to native Word (OMML) maths through pandoc.
"""
from __future__ import annotations

import copy
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH as AL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Pt
from lxml import etree
from PIL import Image

PAPER = Path(__file__).resolve().parent
TEX = PAPER / "tcsst_manuscript.tex"
TEMPLATE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\Users\rishi\Downloads\tcsst-irojournals-paper-template.docx")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else PAPER / "TCSST_manuscript_corrected.docx"
PANDOC = r"C:\Users\rishi\AppData\Local\Pandoc\pandoc.exe"
FONT = "Times New Roman"
TEXT_W_CM = 15.92
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"

# --------------------------------------------------------------------------- inline LaTeX
SKIP_ARG1 = {"vspace", "hspace", "setstretch", "label"}
SKIP_ARG2 = {"fontsize"}
SKIP_NOARG = {"noindent", "par", "centering", "small", "footnotesize", "selectfont", "relax",
              "hfill", "begingroup", "endgroup", "normalsize", "newline", "nopagebreak"}
BOLD = {"textbf"}
ITAL = {"emph", "textit"}
UMLAUT = {"u": "\u00fc", "o": "\u00f6", "a": "\u00e4", "U": "\u00dc", "O": "\u00d6", "A": "\u00c4", "e": "\u00eb"}


def _group(s: str, i: int) -> tuple[str, int]:
    """s[i] == '{' -> (content, index after the matching '}')."""
    depth, j = 0, i
    while j < len(s):
        if s[j] == "\\" and j + 1 < len(s):
            j += 2
            continue
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise ValueError("unbalanced braces: " + s[i:i + 60])


def _merge(segs):
    out = []
    for seg in segs:
        if out and seg[0] == "text" and out[-1][0] == "text" and out[-1][2] == seg[2]:
            out[-1] = ("text", out[-1][1] + seg[1], seg[2])
        else:
            out.append(seg)
    return out


def inline(s: str, fmt: dict | None = None, collapse: bool = True) -> list:
    """LaTeX fragment -> list of (kind, payload, fmt) with kind in text|math|br."""
    fmt = dict(fmt or {})
    if collapse:
        s = re.sub(r"\s+", " ", s)
    segs, i, buf = [], 0, []

    def flush():
        if buf:
            segs.append(("text", "".join(buf), dict(fmt)))
            buf.clear()

    while i < len(s):
        c = s[i]
        if c == "$":
            j = s.index("$", i + 1)
            flush()
            m = s[i + 1:j].strip()
            if re.fullmatch(r"[+\-]?[0-9.]+", m):
                segs.append(("text", m.replace("-", "\u2212"), dict(fmt)))
            elif m == r"\pm":
                segs.append(("text", "\u00b1", dict(fmt)))
            else:
                segs.append(("math", m, dict(fmt)))
            i = j + 1
        elif c == "\\":
            m = re.match(r"\\([A-Za-z]+)\*?", s[i:])
            if m:
                name, i2 = m.group(1), i + m.end()
                if name in BOLD | ITAL:
                    while i2 < len(s) and s[i2] == " ":
                        i2 += 1
                    arg, i3 = _group(s, i2)
                    flush()
                    f2 = dict(fmt)
                    f2["b" if name in BOLD else "i"] = True
                    segs.extend(inline(arg, f2, collapse=False))
                    i = i3
                elif name in SKIP_ARG2:
                    _, i3 = _group(s, i2)
                    _, i3 = _group(s, i3)
                    i = i3
                elif name in SKIP_ARG1:
                    _, i = _group(s, i2)
                elif name in SKIP_NOARG:
                    i = i2
                    while i < len(s) and s[i] == " ":
                        i += 1
                else:
                    raise ValueError(f"unhandled command \\{name} near: {s[i:i + 50]!r}")
            else:
                nxt = s[i + 1] if i + 1 < len(s) else ""
                if nxt in "%&_{}#$":
                    buf.append(nxt)
                    i += 2
                elif nxt == '"':
                    if s[i + 2] == "{":
                        arg, i3 = _group(s, i + 2)
                        buf.append(UMLAUT[arg])
                        i = i3
                    else:
                        buf.append(UMLAUT[s[i + 2]])
                        i += 3
                elif nxt == "\\":
                    flush()
                    segs.append(("br", "", {}))
                    i += 2
                elif nxt == ",":
                    buf.append("\u2009")
                    i += 2
                elif nxt == " ":
                    buf.append(" ")
                    i += 2
                else:
                    raise ValueError(f"unhandled escape near: {s[i:i + 30]!r}")
        elif c in "{}":
            i += 1
        elif c == "~":
            buf.append("\u00a0")
            i += 1
        elif s.startswith("---", i):
            buf.append("\u2014")
            i += 3
        elif s.startswith("--", i):
            buf.append("\u2013")
            i += 2
        elif s.startswith("``", i):
            buf.append("\u201c")
            i += 2
        elif s.startswith("''", i):
            buf.append("\u201d")
            i += 2
        elif c == "'":
            buf.append("\u2019")
            i += 1
        elif c == "`":
            buf.append("\u2018")
            i += 1
        else:
            buf.append(c)
            i += 1
    flush()
    segs = _merge(segs)
    if collapse and segs:
        if segs[0][0] == "text":
            segs[0] = ("text", segs[0][1].lstrip(), segs[0][2])
        if segs[-1][0] == "text":
            segs[-1] = ("text", segs[-1][1].rstrip(), segs[-1][2])
        segs = [s for s in segs if not (s[0] == "text" and s[1] == "")]
    return segs


def plain(segs) -> str:
    return "".join(p for k, p, _ in segs if k == "text")


def math_in(segs) -> set:
    return {p for k, p, _ in segs if k == "math"}


# --------------------------------------------------------------------------- LaTeX source -> blocks
def parse_source():
    src = TEX.read_text(encoding="utf-8-sig")
    title = re.search(r"\\fontsize\{24\}\{24\}\\selectfont\\textbf\s*\{(.*?)\}\s*\\end\{center\}", src, re.S).group(1)
    m0 = re.search(r"\\begin\{(?:onehalfspace|spacing\}\{[\d.]+)\}", src)
    m1 = re.search(r"\\end\{(?:onehalfspace|spacing)\}", src)
    body = src[m0.end():m1.start()]
    i_ref = body.index(r"\section*{\fontsize{12}{12}\selectfont\textbf{References}}")
    i_bio = body.index(r"\section*{\fontsize{12}{12}\selectfont\textbf{Authors' Biographies}}")
    main, refs, bios = body[:i_ref], body[i_ref:i_bio], body[i_bio:]

    envs = []

    def stash(m):
        envs.append((m.group(1), m.group(2) or "", m.group(3)))
        return f"\n\n@@ENV{len(envs) - 1}@@\n\n"

    main = re.sub(r"\\begin\{(figure|table|algorithm|enumerate)\}(\[[^\]]*\])?(.*?)\\end\{\1\}", stash, main, flags=re.S)
    main = re.sub(r"(\\(?:sub)?section\*?\{(?:[^{}]|\{[^{}]*\})*\})", r"\1\n\n", main)
    blocks = []
    for para in re.split(r"\n\s*\n", main):
        p = para.strip()
        if not p:
            continue
        m = re.fullmatch(r"@@ENV(\d+)@@", p)
        if m:
            blocks.append(("env", *envs[int(m.group(1))]))
            continue
        if re.fullmatch(r"(\\vspace\{[^}]*\}\s*)+", p):
            continue
        m = re.fullmatch(r"\\section\*\{(.*)\}", p, re.S)
        if m:
            blocks.append(("sec*", re.sub(r"\\fontsize\{12\}\{12\}\\selectfont|\\textbf", "", m.group(1)).strip("{} ")))
            continue
        m = re.fullmatch(r"\\section\{(.*)\}", p, re.S)
        if m:
            blocks.append(("sec", m.group(1)))
            continue
        m = re.fullmatch(r"\\subsection\{(.*)\}", p, re.S)
        if m:
            blocks.append(("sub", m.group(1)))
            continue
        blocks.append(("para", p))

    # references
    items = re.split(r"\\item\s+", refs[refs.index(r"\begin{enumerate}"):refs.index(r"\end{enumerate}")])[1:]
    ref_items = [re.sub(r"\s+", " ", it).strip() for it in items]
    # biographies
    bio_items = []
    for m in re.finditer(r"\\authorbio\{(\w+)\}\{([^}]*)\}\{", bios):
        arg, _ = _group(bios, m.end() - 1)
        bio_items.append((m.group(1), m.group(2), arg))
    return title, blocks, ref_items, bio_items


# --------------------------------------------------------------------------- math via pandoc
def omml_map(snippets: set) -> dict:
    snippets = sorted(snippets)
    with tempfile.TemporaryDirectory() as tmp:
        md = Path(tmp) / "m.md"
        md.write_text("\n\n".join("$" + s.replace("\\mathrm", "\\text") + "$" for s in snippets) + "\n",
                      encoding="utf-8")
        out = Path(tmp) / "m.docx"
        subprocess.run([PANDOC, str(md), "-f", "markdown+tex_math_dollars", "-t", "docx", "-o", str(out)], check=True)
        with zipfile.ZipFile(out) as z:
            root = etree.fromstring(z.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main", "m": M_NS}
    paras = root.xpath("//w:body/w:p", namespaces=ns)
    assert len(paras) == len(snippets), (len(paras), len(snippets))
    res = {}
    for s, p in zip(snippets, paras):
        om = p.xpath(".//m:oMath", namespaces=ns)
        assert len(om) == 1, s
        for node in om[0].xpath(".//m:t", namespaces=ns):
            if node.text and "\\" in node.text:
                node.text = node.text.replace("\\", "\u2216")      # \setminus
        for nary in om[0].xpath(".//m:nary", namespaces=ns):
            pr = nary.find(f"{{{M_NS}}}naryPr")
            if pr is None:
                pr = etree.SubElement(nary, f"{{{M_NS}}}naryPr")
                nary.insert(0, pr)
            for tag in ("limLoc", "supHide"):
                for e in pr.findall(f"{{{M_NS}}}{tag}"):
                    pr.remove(e)
            loc = etree.Element(f"{{{M_NS}}}limLoc")
            loc.set(f"{{{M_NS}}}val", "subSup")
            hide = etree.Element(f"{{{M_NS}}}supHide")
            hide.set(f"{{{M_NS}}}val", "1")
            chr_ = pr.find(f"{{{M_NS}}}chr")
            pos = (list(pr).index(chr_) + 1) if chr_ is not None else 0
            pr.insert(pos, loc)
            pr.insert(pos + 1, hide)
        txt = "".join(om[0].xpath(".//m:t/text()", namespaces=ns))
        assert not re.search(r"\\[A-Za-z]", txt), ("unconverted macro", s, txt)
        res[s] = om[0]
    return res


# --------------------------------------------------------------------------- docx helpers
class Builder:
    def __init__(self, omml):
        self.doc = docx.Document(str(TEMPLATE))
        body = self.doc.element.body
        for child in list(body):
            if child.tag != qn("w:sectPr"):
                body.remove(child)
        self.omml = omml
        self.after_block = False        # next body paragraph follows a table: 20 pt space before

    # ---- low level
    def _rfonts(self, r, size=None, bold=None, italic=None, sup=False, color=None):
        rPr = r._r.get_or_add_rPr()
        rf = rPr.find(qn("w:rFonts"))
        if rf is None:
            rf = OxmlElement("w:rFonts")
            rPr.insert(0, rf)
        for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            rf.set(qn(a), FONT)
        if size:
            r.font.size = Pt(size)
        if bold is not None:
            r.bold = bold
        if italic is not None:
            r.italic = italic
        if sup:
            r.font.superscript = True
        if color:
            from docx.shared import RGBColor
            r.font.color.rgb = RGBColor.from_string(color)

    def emit(self, par, segs, size=None, bold=False, color=None):
        for kind, payload, fmt in segs:
            if kind == "text":
                for piece, is_url in self._split_urls(payload):
                    if is_url:
                        self._hyperlink(par, piece, 9.5 if len(piece) > 70 else size)
                    else:
                        r = par.add_run(piece)
                        self._rfonts(r, size, bold or fmt.get("b"), fmt.get("i") or None, fmt.get("sup"), color)
            elif kind == "math":
                par._p.append(copy.deepcopy(self.omml[payload]))
            elif kind == "br":
                par.add_run().add_break()

    @staticmethod
    def _split_urls(text):
        out, pos = [], 0
        for m in re.finditer(r"https?://[^\s]+", text):
            url = m.group(0)
            trail = ""
            while url and url[-1] in ".,;)":
                trail = url[-1] + trail
                url = url[:-1]
            if m.start() > pos:
                out.append((text[pos:m.start()], False))
            out.append((url, True))
            if trail:
                out.append((trail, False))
            pos = m.end()
        if pos < len(text):
            out.append((text[pos:], False))
        return out or [(text, False)]

    def _hyperlink(self, par, url, size):
        rid = par.part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
                                 is_external=True)
        h = OxmlElement("w:hyperlink")
        h.set(qn("r:id"), rid)
        h.set(qn("w:history"), "1")
        r = OxmlElement("w:r")
        rPr = OxmlElement("w:rPr")
        st = OxmlElement("w:rStyle")
        st.set(qn("w:val"), "Hyperlink")
        rPr.append(st)
        rf = OxmlElement("w:rFonts")
        for a in ("w:ascii", "w:hAnsi", "w:cs"):
            rf.set(qn(a), FONT)
        rPr.append(rf)
        if size:
            for tag in ("w:sz", "w:szCs"):
                e = OxmlElement(tag)
                e.set(qn("w:val"), str(int(size * 2)))
                rPr.append(e)
        r.append(rPr)
        t = OxmlElement("w:t")
        t.text = url
        t.set(qn("xml:space"), "preserve")
        r.append(t)
        h.append(r)
        par._p.append(h)

    def para(self, style=None, align=None, before=None, after=None, line=1.0, first=None, left=None,
             hanging=None, keep_next=False, keep_lines=False):
        p = self.doc.add_paragraph(style=style)
        pf = p.paragraph_format
        pf.line_spacing = line
        if align is not None:
            p.alignment = align
        if before is not None:
            pf.space_before = Pt(before)
        if after is not None:
            pf.space_after = Pt(after)
        if first is not None:
            pf.first_line_indent = Cm(first)
        if left is not None:
            pf.left_indent = Cm(left)
        if hanging is not None:
            pf.left_indent = Cm(hanging)
            pf.first_line_indent = Cm(-hanging)
        if keep_next:
            pf.keep_with_next = True
        if keep_lines:
            pf.keep_together = True
        return p

    def pending_before(self):
        if self.after_block:
            self.after_block = False
            return 20
        return None

    # ---- title block
    def title_block(self, title):
        p = self.para(None, AL.CENTER, after=24)
        self._rfonts(p.add_run(re.sub(r"\s+", " ", title).replace("'", "\u2019")), 24, True)
        names = [("Rishi Jayanath A", "1"), ("Adwaith Sajikumar", "1"), ("G.\u00a0Naveen\u00a0Sundar", "2"), ("D.\u00a0Narmadha", "3")]
        p = self.para(None, None, after=6)
        for k, (n, a) in enumerate(names):
            self._rfonts(p.add_run(n), 16, True)
            self._rfonts(p.add_run(a), 16, True, sup=True)
            self._rfonts(p.add_run(", "), 16, True)
        self._rfonts(p.add_run("for the Alzheimer\u2019s Disease Neuroimaging Initiative"), 16, True)
        self._rfonts(p.add_run("*"), 16, True, sup=True)
        affs = [("1", "Department of Computer Science and Engineering, Karunya Institute of Technology and Sciences, Coimbatore, India"),
                ("2", "Division of Computer Science and Engineering, Karunya Institute of Technology and Sciences, Coimbatore, Tamil Nadu, India"),
                ("3", "Division of AIML, Karunya Institute of Technology and Sciences, Coimbatore, Tamil Nadu, India")]
        for k, (n, t) in enumerate(affs):
            p = self.para(None, AL.JUSTIFY, after=8 if k == len(affs) - 1 else 0, line=1.08)
            self._rfonts(p.add_run(n), 11, sup=True)
            self._rfonts(p.add_run(t), 11)
        p = self.para(None, AL.LEFT, line=1.0)
        self._rfonts(p.add_run("E-mail: "), 9, True)
        mails = [("1", "rishijayanath@karunya.edu.in"), ("1", "adwaithsajikumar@karunya.edu.in"),
                 ("2", "naveensundar@karunya.edu"), ("3", "narmadha@karunya.edu")]
        for k, (n, m) in enumerate(mails):
            self._rfonts(p.add_run(n), 9, sup=True)
            self._rfonts(p.add_run(m), 9)
            if k < len(mails) - 1:
                self._rfonts(p.add_run(", "), 9)
        note = ("*Data used in preparation of this article were obtained from the Alzheimer\u2019s Disease "
                "Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). As such, the investigators within "
                "the ADNI contributed to the design and implementation of ADNI and/or provided data but did not "
                "participate in analysis or writing of this report. A complete listing of ADNI investigators can "
                "be found at: ")
        q = self.para(None, AL.LEFT, before=6, line=1.0)
        self._rfonts(q.add_run(note), 9)
        self._hyperlink(q, "http://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf", 9)

    # ---- headings / paragraphs
    def unnumbered_heading(self, text):
        p = self.para(None, AL.JUSTIFY, before=20, keep_next=True)
        self._rfonts(p.add_run(text), 12, True)

    def heading(self, text):
        p = self.para("IRO Heading", keep_next=True)
        self._rfonts(p.add_run(text), 12, True)
        self.after_block = False

    def subheading(self, num, text):
        p = self.para("List Paragraph", AL.JUSTIFY, before=10, keep_next=True)
        pf = p.paragraph_format
        pf.left_indent = Cm(0.75)
        pf.first_line_indent = Cm(-0.75)
        self._rfonts(p.add_run(f"{num}\t"), 12, True)
        self.emit(p, inline(text, {"b": True}), 12, True)
        self.after_block = False

    def body_par(self, segs, abstract=False, keywords=False, size=12):
        if abstract:
            p = self.para(None, AL.JUSTIFY, after=0)
            self.emit(p, segs, 12)
        elif keywords:
            p = self.para(None, AL.JUSTIFY, before=5, after=0)
            self.emit(p, segs, 12)
        else:
            p = self.para("IRO Parag", before=self.pending_before())
            if "http" in plain(segs):
                p.alignment = AL.LEFT
            self.emit(p, segs, size)
        return p

    def list_item(self, label, segs):
        p = self.para("IRO Parag", AL.JUSTIFY, after=4, hanging=1.0, first=None)
        p.paragraph_format.left_indent = Cm(1.0)
        p.paragraph_format.first_line_indent = Cm(-1.0)
        self._rfonts(p.add_run(f"{label}\t"), 12)
        self.emit(p, segs, 12)

    # ---- figures
    def figure(self, images, caption_segs, labels=None):
        n = len(images)
        for k, (path, wcm) in enumerate(images):
            p = self.para(None, AL.CENTER, before=20 if k == 0 else 8, after=5 if n == 1 else 2,
                          keep_next=True)
            p.add_run().add_picture(str(path), width=Cm(wcm))
            if n > 1:
                q = self.para(None, AL.CENTER, before=0, after=4, keep_next=True)
                self._rfonts(q.add_run(f"({'ab'[k]})"), 12, True)
        cap = self.para(None, AL.CENTER, before=5, after=20, keep_lines=True)
        self.emit(cap, caption_segs, 12)
        for r in cap.runs[:1]:
            pass
        self.after_block = False

    # ---- tables
    def _tbl_xml(self, widths_cm):
        total = int(sum(widths_cm) / 2.54 * 1440)
        grid = "".join(f'<w:gridCol w:w="{int(w / 2.54 * 1440)}"/>' for w in widths_cm)
        return total, grid

    def table(self, rows, widths_cm, aligns, header=True, caption=None, size=12):
        if caption:
            p = self.para(None, AL.CENTER, before=20, after=5, keep_next=True, keep_lines=True)
            self.emit(p, caption, 12)
        nrows, ncols = len(rows), len(widths_cm)
        t = self.doc.add_table(rows=nrows, cols=ncols)
        t.autofit = False
        tblPr = t._tbl.tblPr
        for el in list(tblPr):
            if el.tag in (qn("w:tblStyle"), qn("w:tblW"), qn("w:tblBorders"), qn("w:tblLayout"), qn("w:jc")):
                tblPr.remove(el)
        total, _ = self._tbl_xml(widths_cm)
        tblPr.append(parse_xml(f'<w:tblW {nsdecls("w")} w:w="{total}" w:type="dxa"/>'))
        tblPr.append(parse_xml(f'<w:jc {nsdecls("w")} w:val="center"/>'))
        b = "".join(f'<w:{s} w:val="single" w:sz="8" w:space="0" w:color="000000"/>'
                    for s in ("top", "left", "bottom", "right", "insideH", "insideV"))
        tblPr.append(parse_xml(f'<w:tblBorders {nsdecls("w")}>{b}</w:tblBorders>'))
        tblPr.append(parse_xml(f'<w:tblLayout {nsdecls("w")} w:type="fixed"/>'))
        tblPr.append(parse_xml(
            f'<w:tblCellMar {nsdecls("w")}><w:top w:w="0" w:type="dxa"/><w:left w:w="100" w:type="dxa"/>'
            f'<w:bottom w:w="0" w:type="dxa"/><w:right w:w="100" w:type="dxa"/></w:tblCellMar>'))
        for ci, w in enumerate(widths_cm):
            t.columns[ci].width = Cm(w)
        for ri, row in enumerate(rows):
            tr = t.rows[ri]
            trPr = tr._tr.get_or_add_trPr()
            trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
            if header and ri == 0:
                trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
            ci = 0
            for cell_spec in row:
                span, segs = cell_spec
                cell = tr.cells[ci]
                if span > 1:
                    cell = cell.merge(tr.cells[ci + span - 1])
                cell.width = Cm(sum(widths_cm[ci:ci + span]))
                tcPr = cell._tc.get_or_add_tcPr()
                tcPr.append(parse_xml(f'<w:vAlign {nsdecls("w")} w:val="center"/>'))
                p = cell.paragraphs[0]
                p.style = self.doc.styles["IRO Parag"]
                pf = p.paragraph_format
                pf.first_line_indent = Cm(0)
                pf.line_spacing = 1.0
                pf.space_before = Pt(3)
                pf.space_after = Pt(3)
                left = span > 1 or aligns[ci] == "l"
                p.alignment = AL.LEFT if left else AL.CENTER
                is_head = header and ri == 0
                if nrows <= 8 and ri < nrows - 1:
                    pf.keep_with_next = True          # keep short tables on one page
                self.emit(p, segs, size, bold=is_head)
                ci += span
        self.after_block = True
        return t

    def algorithm(self, title_segs, lines):
        t = self.doc.add_table(rows=2, cols=1)
        t.autofit = False
        # move the (empty) spacer paragraph away: the table sits directly after it
        tblPr = t._tbl.tblPr
        for el in list(tblPr):
            if el.tag in (qn("w:tblStyle"), qn("w:tblW"), qn("w:tblLayout")):
                tblPr.remove(el)
        w_twips = int(TEXT_W_CM / 2.54 * 1440)
        tblPr.append(parse_xml(f'<w:tblW {nsdecls("w")} w:w="{w_twips}" w:type="dxa"/>'))
        tblPr.append(parse_xml(f'<w:tblLayout {nsdecls("w")} w:type="fixed"/>'))
        t.columns[0].width = Cm(TEXT_W_CM)
        for ri, bords in ((0, ("top", "bottom")), (1, ("bottom",))):
            if ri == 0:
                t.rows[ri]._tr.get_or_add_trPr().append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
            tcPr = t.rows[ri].cells[0]._tc.get_or_add_tcPr()
            tcPr.append(parse_xml(
                f'<w:tcBorders {nsdecls("w")}>' + "".join(f'<w:{s} w:val="single" w:sz="8" w:space="0" w:color="000000"/>' for s in bords)
                + "</w:tcBorders>"))
        c0, c1 = t.rows[0].cells[0], t.rows[1].cells[0]
        pt = c0.paragraphs[0]
        pt.style = self.doc.styles["IRO Parag"]
        pt.paragraph_format.first_line_indent = Cm(0)
        pt.paragraph_format.line_spacing = 1.0
        pt.paragraph_format.space_before = Pt(3)
        pt.paragraph_format.space_after = Pt(3)
        pt.alignment = AL.LEFT
        pt.paragraph_format.keep_with_next = True
        self._rfonts(pt.add_run("Algorithm 1. "), 12, True)
        self.emit(pt, title_segs, 12, True)
        first = True
        for level, label, segs in lines:
            q = c1.paragraphs[0] if first else c1.add_paragraph()
            first = False
            q.style = self.doc.styles["IRO Parag"]
            pf = q.paragraph_format
            pf.line_spacing = 1.0
            pf.space_before = Pt(2)
            pf.space_after = Pt(2)
            base = 0.9 + 0.7 * level
            pf.left_indent = Cm(base + (0.0 if label else 0))
            pf.first_line_indent = Cm(-0.9) if label else Cm(0)
            if not label:
                pf.left_indent = Cm(0.2)
                pf.first_line_indent = Cm(0)
            q.alignment = AL.LEFT
            if label:
                self._rfonts(q.add_run(label + "\t"), 12)
            self.emit(q, segs, 12)
        self.after_block = True

    def bio(self, photo, name, text):
        t = self.doc.add_table(rows=1, cols=2)
        t.autofit = False
        tblPr = t._tbl.tblPr
        for el in list(tblPr):
            if el.tag in (qn("w:tblStyle"), qn("w:tblW"), qn("w:tblLayout")):
                tblPr.remove(el)
        tblPr.append(parse_xml(f'<w:tblW {nsdecls("w")} w:w="{int(TEXT_W_CM / 2.54 * 1440)}" w:type="dxa"/>'))
        tblPr.append(parse_xml(f'<w:tblLayout {nsdecls("w")} w:type="fixed"/>'))
        t.columns[0].width = Cm(3.0)
        t.columns[1].width = Cm(TEXT_W_CM - 3.0)
        t.rows[0]._tr.get_or_add_trPr().append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        c0, c1 = t.rows[0].cells
        c0.width, c1.width = Cm(3.0), Cm(TEXT_W_CM - 3.0)
        p = c0.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        p.add_run().add_picture(str(photo), width=Cm(2.2))
        q = c1.paragraphs[0]
        q.style = self.doc.styles["IRO Parag"]
        q.paragraph_format.first_line_indent = Cm(0)
        q.paragraph_format.line_spacing = 1.0
        q.paragraph_format.space_after = Pt(0)
        q.alignment = AL.LEFT
        self._rfonts(q.add_run(name + " "), 10, True)
        self.emit(q, inline(text), 10)
        sp = self.para(None, None, before=0, after=0)
        sp.paragraph_format.line_spacing = Pt(5)

    def reference(self, segs):
        p = self.para("List Paragraph", AL.LEFT)
        pPr = p._p.get_or_add_pPr()
        pPr.append(parse_xml(f'<w:numPr {nsdecls("w")}><w:ilvl w:val="0"/><w:numId w:val="31"/></w:numPr>'))
        pf = p.paragraph_format
        pf.left_indent = Cm(1.5)
        pf.first_line_indent = Cm(-1.5)
        pPr.append(parse_xml(f'<w:tabs {nsdecls("w")}><w:tab w:val="left" w:pos="850"/></w:tabs>'))
        pPr.append(parse_xml(f'<w:rPr {nsdecls("w")}><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
                             f'w:cs="Times New Roman"/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>'))
        self.emit(p, segs, 10, color="222222")


# --------------------------------------------------------------------------- tables / figures / algorithm from LaTeX
def parse_table(raw):
    cap = re.search(r"\\textbf\{(Table \d+\.)\}(.*?)\\\\\\vspace\{5pt\}", raw, re.S)
    caption = inline(r"\textbf{" + cap.group(1) + "}" + cap.group(2))
    tab = re.search(r"\\begin\{tabular\}\{(.*?)\}\n(.*?)\\end\{tabular\}", raw, re.S)
    spec, content = tab.group(1), tab.group(2)
    cols = re.findall(r"([Mp])\{([\d.]+)cm\}", spec)
    aligns = ["l" if k == "p" else "c" for k, _ in cols]
    rows = []
    for line in re.split(r"\\\\\s*(?:\\hline)?", content.replace("\\hline", "")):
        line = line.strip()
        if not line:
            continue
        mc = re.fullmatch(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}", line, re.S)
        if mc:
            rows.append([(int(mc.group(1)), inline(mc.group(2)))])
        else:
            rows.append([(1, inline(c)) for c in re.split(r"(?<!\\)&", line)])
    return caption, rows, aligns


TABLE_WIDTHS = {  # cm; chosen to fill the 15.9 cm text width
    1: [4.6, 1.8, 2.6, 6.9],
    2: [8.2, 2.5, 2.5, 2.5],
    3: [6.1, 2.0, 2.6, 2.6, 2.6],
    4: [5.2, 2.0, 2.9, 2.9, 2.9],
    5: [3.9, 4.0, 4.0, 4.0],
    6: [6.8, 2.2, 3.4, 3.4],
}
DEFAULT_MAX_W = 15.9
FIG_DEFER = {2: 2, 8: 1, 1: 2, 9: 1}          # figure number -> number of body paragraphs to place before it
IMG_MAX_W = {"fig_algorithm_flow.png": 14.6, 
             "fig_final_test_forest.png": 14.6}


def parse_figure(raw):
    imgs = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", raw)
    cap = re.search(r"(\\textbf\{Figure \d+\.\}.*?)(?:\\vspace\{\d+pt\}\s*)?$", raw, re.S)
    caption = inline(cap.group(1).strip())
    return imgs, caption


def img_width(name):
    px = Image.open(PAPER / name).size
    w = px[0] / 300 * 2.54
    h = px[1] / 300 * 2.54
    w = min(w, IMG_MAX_W.get(name, DEFAULT_MAX_W))
    if h * (w / (px[0] / 300 * 2.54)) > 18.5:
        w = w * 18.5 / (h * (w / (px[0] / 300 * 2.54)))
    return round(w, 2)


def parse_algorithm(raw):
    cap = re.search(r"\\caption\{(.*)\}\s*\\begin\{algorithmic\}", raw, re.S).group(1)
    body = re.search(r"\\begin\{algorithmic\}\[1\](.*)\\end\{algorithmic\}", raw, re.S).group(1)
    toks = re.split(r"(?=\\(?:Require|Ensure|State|If|EndIf|ForAll|EndFor|Return)\b)", body)
    lines, level, n = [], 0, 0

    def kw(t):
        return ("text", t, {"b": True})

    for tok in toks:
        tok = tok.strip()
        if not tok:
            continue
        comment = []
        k = tok.find("\\Comment{")
        if k >= 0:
            arg, end = _group(tok, k + len("\\Comment"))
            comment = [("text", "   \u25b7 ", {"i": True})] + inline(arg, {"i": True})
            tok = tok[:k] + tok[end:]
        name = re.match(r"\\(\w+)", tok).group(1)
        rest = tok[len(name) + 1:].strip()
        if name in ("Require", "Ensure"):
            lines.append((0, "", [kw(name + ": ")] + inline(rest)))
        elif name == "If":
            cond, _ = _group(rest, rest.index("{"))
            n += 1
            lines.append((level, f"{n}:", [kw("if ")] + inline(cond) + [kw(" then")] + comment))
            level += 1
        elif name == "ForAll":
            cond, _ = _group(rest, rest.index("{"))
            n += 1
            lines.append((level, f"{n}:", [kw("for all ")] + inline(cond) + [kw(" do")] + comment))
            level += 1
        elif name == "EndIf":
            level -= 1
            n += 1
            lines.append((level, f"{n}:", [kw("end if")]))
        elif name == "EndFor":
            level -= 1
            n += 1
            lines.append((level, f"{n}:", [kw("end for")]))
        elif name == "Return":
            n += 1
            lines.append((level, f"{n}:", [kw("return ")] + inline(rest) + comment))
        else:  # State
            if not rest:
                continue
            n += 1
            lines.append((level, f"{n}:", inline(rest) + comment))
    return inline(cap), lines


def ordinal_label(arg, k):
    return ["(i)", "(ii)", "(iii)", "(iv)", "(v)"][k]


# --------------------------------------------------------------------------- schema element order
ORDERS = {
    "pPr": ["pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr", "widowControl", "numPr",
            "suppressLineNumbers", "pBdr", "shd", "tabs", "suppressAutoHyphens", "kinsoku", "wordWrap",
            "overflowPunct", "topLinePunct", "autoSpaceDE", "autoSpaceDN", "bidi", "adjustRightInd",
            "snapToGrid", "spacing", "ind", "contextualSpacing", "mirrorIndents", "suppressOverlap", "jc",
            "textDirection", "textAlignment", "textboxTightWrap", "outlineLvl", "divId", "cnfStyle", "rPr",
            "sectPr", "pPrChange"],
    "tblPr": ["tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize", "tblStyleColBandSize",
              "tblW", "jc", "tblCellSpacing", "tblInd", "tblBorders", "shd", "tblLayout", "tblCellMar",
              "tblLook", "tblCaption", "tblDescription"],
    "tcPr": ["cnfStyle", "tcW", "gridSpan", "hMerge", "vMerge", "tcBorders", "shd", "noWrap", "tcMar",
             "textDirection", "tcFitText", "vAlign", "hideMark"],
    "naryPr": ["chr", "limLoc", "grow", "subHide", "supHide", "ctrlPr"],
}


def normalise_order(root):
    for el in root.iter():
        name = etree.QName(el).localname if isinstance(el.tag, str) else None
        if name in ORDERS:
            order = ORDERS[name]
            kids = list(el)
            kids.sort(key=lambda c: order.index(etree.QName(c).localname)
                      if etree.QName(c).localname in order else len(order))
            for c in kids:
                el.remove(c)
            for c in kids:
                el.append(c)


# --------------------------------------------------------------------------- main
def main():
    title, blocks, refs, bios = parse_source()

    # pass 1: gather math
    math = set()
    parsed = []
    for blk in blocks:
        kind = blk[0]
        if kind == "para":
            segs = inline(blk[1])
            math |= math_in(segs)
            parsed.append(("para", segs))
        elif kind == "env":
            _, name, arg, raw = blk
            if name == "table":
                cap, rows, aligns = parse_table(raw)
                math |= math_in(cap)
                for r in rows:
                    for _, s in r:
                        math |= math_in(s)
                parsed.append(("table", cap, rows, aligns))
            elif name == "figure":
                imgs, cap = parse_figure(raw)
                math |= math_in(cap)
                parsed.append(("figure", imgs, cap))
            elif name == "algorithm":
                cap, lines = parse_algorithm(raw)
                math |= math_in(cap)
                for _, _, s in lines:
                    math |= math_in(s)
                parsed.append(("algorithm", cap, lines))
            elif name == "enumerate":
                items = [inline(re.sub(r"\s+", " ", it).strip()) for it in re.split(r"\\item\s+", raw)[1:]]
                for s in items:
                    math |= math_in(s)
                parsed.append(("enum", items))
        else:
            parsed.append(blk)
    ref_segs = [inline(r) for r in refs]
    omml = omml_map(math)

    b = Builder(omml)
    b.title_block(title)
    sec_no, sub_no = 0, 0
    in_abstract = False
    small_text = False
    pending = []                      # [remaining paragraphs, figure item]

    def put_figure(it):
        _, imgs, cap = it
        b.figure([(PAPER / n, img_width(n)) for n in imgs], cap)

    def flush_all():
        while pending:
            put_figure(pending.pop(0)[1])

    def release():
        while pending and pending[0][0] <= 0:
            put_figure(pending.pop(0)[1])

    for item in parsed:
        kind = item[0]
        if kind == "figure":
            num = int(re.search(r"Figure (\d+)", plain(item[2])).group(1))
            if FIG_DEFER.get(num) or pending:
                pending.append([FIG_DEFER.get(num, 0), item])    # keep figure order
                release()
            else:
                put_figure(item)
            continue
        if kind == "sec*":
            flush_all()
            text = item[1]
            b.unnumbered_heading(text)
            in_abstract = text == "Abstract"
            small_text = text == "Acknowledgement"
        elif kind == "sec":
            flush_all()
            sec_no += 1
            sub_no = 0
            text = item[1].replace("Materials and Methods", "Proposed Work")
            b.heading(text)
        elif kind == "sub":
            sub_no += 1
            b.subheading(f"{sec_no}.{sub_no}", item[1])
        elif kind == "para":
            segs = item[1]
            if plain(segs).lstrip().startswith("Keywords:"):
                b.body_par(segs, keywords=True)
                in_abstract = False
            else:
                b.body_par(segs, abstract=in_abstract, size=10 if small_text else 12)
                for entry in pending:
                    entry[0] -= 1
                release()
        elif kind == "table":
            _, cap, rows, aligns = item
            num = int(re.search(r"Table (\d+)", plain(cap)).group(1))
            b.table(rows, TABLE_WIDTHS[num], aligns, caption=cap, size=10 if num in (1, 2, 5) else 12)
        elif kind == "algorithm":
            b.algorithm(item[1], item[2])
        elif kind == "enum":
            for k, segs in enumerate(item[1]):
                b.list_item(ordinal_label(None, k), segs)
    flush_all()

    b.unnumbered_heading("References")
    for s in ref_segs:
        b.reference(s)
    b.unnumbered_heading("Authors' Biographies")
    photos = {"rishi": "rishi.jpg", "adwaith": "adwaith.jpg", "naveen": "naveen.jpg", "narmadha": "narmadha.jpg"}
    for key, name, text in bios:
        b.bio(PAPER / "authors" / photos[key], name, text)

    cp = b.doc.core_properties
    cp.title = re.sub(r"\s+", " ", title).replace("\\", "")
    cp.author = "Rishi Jayanath A, Adwaith Sajikumar, G. Naveen Sundar, D. Narmadha"
    normalise_order(b.doc.element)
    b.doc.save(str(OUT))
    print("saved", OUT)


if __name__ == "__main__":
    main()


