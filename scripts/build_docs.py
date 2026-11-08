"""Build the report PDF and the slide deck from two small Markdown sources.

    python scripts/build_docs.py      # docs/report.md -> docs/report.pdf, docs/slides.md -> docs/slides.pptx

report.md: '# ' title, '## ' / '### ' headings, paragraphs, '- ' bullets, '|' tables (first row = header),
           '![caption](path){width}' images (width in cm, optional), '<!-- pagebreak -->', '**bold**', '*italic*',
           `code`.
slides.md: slides separated by a line '---'. First line '# title'; '- ' bullets (two-space indent = level 2);
           '![](path)' one image (placed right of the bullets, or full width if there are no bullets);
           '|' table; 'Notes:' starts the speaker notes (rest of the slide)."""
import os, re, sys
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ACCENT = colors.HexColor("#1f5fbf")

def register_fonts():
    """DejaVu (ships with matplotlib) renders Arabic letters, arrows, ±, σ, ⊙ that Helvetica cannot."""
    import matplotlib
    d = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
    for name, f in [("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"),
                    ("DV-BI", "DejaVuSans-BoldOblique.ttf"), ("DVM", "DejaVuSansMono.ttf")]:
        pdfmetrics.registerFont(TTFont(name, os.path.join(d, f)))
    pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

def inline(t):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"`([^`]+)`", r'<font face="DVM" size="8">\1</font>', t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="#1f5fbf">\1</link>', t)
    return t

def styles():
    ss = getSampleStyleSheet()
    base = ParagraphStyle("body", parent=ss["BodyText"], fontName="DV", fontSize=8.6, leading=11.2,
                          alignment=TA_JUSTIFY, spaceAfter=3.5)
    return dict(
        body=base,
        title=ParagraphStyle("title", parent=base, fontName="DV-B", fontSize=14.5, leading=18, alignment=TA_CENTER,
                             spaceAfter=2),
        sub=ParagraphStyle("sub", parent=base, fontSize=8.4, alignment=TA_CENTER, textColor=colors.HexColor("#444"),
                           spaceAfter=6),
        h2=ParagraphStyle("h2", parent=base, fontName="DV-B", fontSize=10.5, leading=13, textColor=ACCENT,
                          spaceBefore=6, spaceAfter=2.5, alignment=0),
        h3=ParagraphStyle("h3", parent=base, fontName="DV-B", fontSize=9, leading=11.5, spaceBefore=3, spaceAfter=1.5,
                          alignment=0),
        cell=ParagraphStyle("cell", parent=base, fontSize=7.2, leading=8.8, alignment=0, spaceAfter=0),
        cellb=ParagraphStyle("cellb", parent=base, fontName="DV-B", fontSize=7.2, leading=8.8, alignment=0,
                             spaceAfter=0),
        cap=ParagraphStyle("cap", parent=base, fontSize=7.4, leading=9, textColor=colors.HexColor("#444"),
                           alignment=TA_CENTER, spaceAfter=5),
        bullet=ParagraphStyle("bullet", parent=base, spaceAfter=1.2),
    )

def table_flow(rows, S, width):
    data = [[Paragraph(inline(c), S["cellb"] if i == 0 else S["cell"]) for c in r] for i, r in enumerate(rows)]
    ncol = len(rows[0])
    # first column wider when it holds labels
    w0 = min(0.34, max(0.18, max(len(r[0]) for r in rows) / 120))
    cw = [width * w0] + [width * (1 - w0) / (ncol - 1)] * (ncol - 1) if ncol > 1 else [width]
    t = Table(data, colWidths=cw, repeatRows=1)
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black), ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 2.5), ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6fa")])]))
    return t

def parse_table(lines):
    rows = []
    for l in lines:
        cells = [c.strip() for c in l.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
            continue
        rows.append(cells)
    return rows

def md_to_pdf(src, dst):
    register_fonts()
    S = styles()
    W = A4[0] - 3.2 * cm
    lines = open(src, encoding="utf-8").read().split("\n")
    story, i, para = [], 0, []
    base = os.path.dirname(os.path.abspath(src))

    def flush():
        if para:
            story.append(Paragraph(inline(" ".join(para)), S["body"])); para.clear()

    while i < len(lines):
        l = lines[i]
        s = l.strip()
        if not s:
            flush(); i += 1; continue
        if s == "<!-- pagebreak -->":
            flush(); story.append(PageBreak()); i += 1; continue
        if s.startswith("<!--"):
            i += 1; continue
        if s.startswith("# "):
            flush(); story.append(Paragraph(inline(s[2:]), S["title"])); i += 1
            sub = []
            while i < len(lines) and lines[i].strip().startswith("> "):
                sub.append(inline(lines[i].strip()[2:])); i += 1
            if sub:
                story.append(Paragraph("<br/>".join(sub), S["sub"]))
            continue
        if s.startswith("## "):
            flush(); story.append(Paragraph(inline(s[3:]), S["h2"])); i += 1; continue
        if s.startswith("### "):
            flush(); story.append(Paragraph(inline(s[4:]), S["h3"])); i += 1; continue
        m = re.match(r"!\[(.*?)\]\((.+?)\)(\{([\d.]+)\})?", s)
        if m:
            flush()
            p = os.path.join(base, m.group(2))
            from PIL import Image as PI
            w_px, h_px = PI.open(p).size
            w = float(m.group(4)) * cm if m.group(4) else W
            img = Image(p, width=w, height=w * h_px / w_px)
            story.append(KeepTogether([img, Paragraph(inline(m.group(1)), S["cap"])] if m.group(1) else [img]))
            i += 1; continue
        if s.startswith("|"):
            flush(); tl = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tl.append(lines[i]); i += 1
            cap = None
            if i < len(lines) and lines[i].strip().startswith("Table"):
                cap = lines[i].strip(); i += 1
            parts = [table_flow(parse_table(tl), S, W)]
            if cap:
                parts.append(Paragraph(inline(cap), S["cap"]))
            else:
                parts.append(Spacer(1, 4))
            story.append(KeepTogether(parts)); continue
        if re.match(r"(- |\d+\. )", s):
            flush(); items = []; numbered = s[0].isdigit()
            while i < len(lines) and re.match(r"\s*(- |\d+\. )", lines[i]):
                t = re.sub(r"^\s*(- |\d+\. )", "", lines[i]); i += 1
                while i < len(lines) and lines[i].startswith("  ") and not re.match(r"\s*(- |\d+\. )", lines[i]):
                    t += " " + lines[i].strip(); i += 1
                items.append(ListItem(Paragraph(inline(t), S["bullet"]), leftIndent=10))
            story.append(ListFlowable(items, bulletType="1" if numbered else "bullet", start="1" if numbered else "•",
                                      leftIndent=10, bulletFontName="DV", bulletFontSize=7.5))
            continue
        para.append(s); i += 1
    flush()

    def deco(c, d):
        c.saveState(); c.setFont("DV", 7); c.setFillColor(colors.HexColor("#666"))
        c.drawCentredString(A4[0] / 2, 0.8 * cm, str(d.page)); c.restoreState()

    doc = SimpleDocTemplate(dst, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.3 * cm,
                            bottomMargin=1.3 * cm, title="EN→Darija seq2seq: comparing LSTM cells",
                            author="", subject="Advanced NLP lab project")
    doc.build(story, onFirstPage=deco, onLaterPages=deco)
    from pypdf import PdfReader  # optional page count
    print("wrote", dst, len(PdfReader(dst).pages), "pages")

def md_to_pptx(src, dst):
    from pptx import Presentation
    from pptx.util import Cm, Pt
    from pptx.dml.color import RGBColor
    base = os.path.dirname(os.path.abspath(src))
    prs = Presentation(); prs.slide_width, prs.slide_height = Cm(33.867), Cm(19.05)
    blank = prs.slide_layouts[6]
    chunks = re.split(r"\n---\s*\n", open(src, encoding="utf-8").read().strip())
    for n, ch in enumerate(chunks):
        lines = ch.strip().split("\n")
        title = lines[0].lstrip("# ").strip()
        notes, body, img, table, sub = [], [], None, [], []
        in_notes = False
        for l in lines[1:]:
            if l.strip().startswith("Notes:"):
                in_notes = True; notes.append(l.split("Notes:", 1)[1].strip()); continue
            if in_notes:
                notes.append(l); continue
            m = re.match(r"\s*!\[.*?\]\((.+?)\)", l)
            if m:
                img = os.path.join(base, m.group(1)); continue
            if l.strip().startswith("|"):
                table.append(l); continue
            if l.strip().startswith("> "):
                sub.append(l.strip()[2:]); continue
            if l.strip():
                body.append(l)
        s = prs.slides.add_slide(blank)
        W, H = prs.slide_width, prs.slide_height
        first = n == 0
        tb = s.shapes.add_textbox(Cm(1.5), Cm(5.5) if first else Cm(0.9), W - Cm(3), Cm(3) if first else Cm(2))
        p = tb.text_frame.paragraphs[0]; p.text = title; tb.text_frame.word_wrap = True
        p.font.size = Pt(34 if first else 26); p.font.bold = True; p.font.color.rgb = RGBColor(0x1f, 0x5f, 0xbf)
        if not first:
            ln = s.shapes.add_shape(1, Cm(1.5), Cm(2.75), Cm(3), Cm(0.12))
            ln.fill.solid(); ln.fill.fore_color.rgb = RGBColor(0x1f, 0x5f, 0xbf); ln.line.fill.background()
        top = Cm(9) if first else Cm(3.3)
        if sub:
            sb = s.shapes.add_textbox(Cm(1.5), top, W - Cm(3), Cm(3)); sb.text_frame.word_wrap = True
            for k, t in enumerate(sub):
                pp = sb.text_frame.paragraphs[0] if k == 0 else sb.text_frame.add_paragraph()
                pp.text = t; pp.font.size = Pt(16 if first else 14); pp.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
            top = int(top + Cm(1.2 * len(sub)))
        text_w = W - Cm(3)
        stacked = bool(img and table and not body)          # table on top, figure below, both full width
        if img and (body or table) and not stacked:
            text_w = int((W - Cm(3)) * 0.42)
        if body:
            bx = s.shapes.add_textbox(Cm(1.5), top, text_w, H - top - Cm(1)); tf = bx.text_frame; tf.word_wrap = True
            for k, l in enumerate(body):
                lvl = 1 if l.startswith("  ") else 0
                t = re.sub(r"^\s*-\s*", "", l)
                pp = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
                pp.level = lvl; pp.space_after = Pt(7 if lvl == 0 else 4)
                parts = [("• " if lvl == 0 else "– ", False)] + \
                    [(seg, i % 2 == 1) for i, seg in enumerate(t.split("**")) if seg]
                for seg, bold in parts:
                    r = pp.add_run(); r.text = seg; r.font.bold = bold
                    r.font.size = Pt(19 if lvl == 0 else 15)
                    r.font.color.rgb = RGBColor(0x22, 0x22, 0x22) if lvl == 0 else RGBColor(0x55, 0x55, 0x55)
        if table:
            rows = parse_table(table)
            body_h = int(Cm(1.05) * len(body) + Cm(0.4)) if body and not img else 0
            ttop = int(top + body_h)
            tw = text_w if (img and not stacked) else W - Cm(3)
            rh = Cm(1.0) if len(rows) < 7 else Cm(0.8)
            shp = s.shapes.add_table(len(rows), len(rows[0]), Cm(1.5), ttop, tw, int(rh * len(rows)))
            for r_i, row in enumerate(rows):
                for c_i, val in enumerate(row):
                    cell = shp.table.cell(r_i, c_i); cell.text = re.sub(r"\*\*(.+?)\*\*", r"\1", val)
                    for pp in cell.text_frame.paragraphs:
                        pp.font.size = Pt(15 if len(rows) < 7 else 12); pp.font.bold = r_i == 0 or val.startswith("**")
            table_bottom = ttop + int(rh * len(rows))
        if img:
            from PIL import Image as PI
            w_px, h_px = PI.open(img).size
            if stacked:
                itop = int(table_bottom + Cm(0.6)); maxw = W - Cm(3); maxh = H - itop - Cm(1.0)
            elif body or table:
                itop = top; maxw = W - (Cm(1.5) + text_w + Cm(0.8)) - Cm(1.2); maxh = H - top - Cm(1.0)
            else:
                itop = top; maxw = W - Cm(3); maxh = H - top - Cm(1.0)
            w = int(min(maxw, maxh * w_px / h_px)); h = int(w * h_px / w_px)
            if stacked or not (body or table):
                left = int((W - w) / 2)
            else:
                left = int(Cm(1.5) + text_w + Cm(0.8) + (maxw - w) / 2)
                itop = int(top + max(0, (H - top - Cm(1.0) - h) / 2))
            s.shapes.add_picture(img, left, itop, w, h)
        if notes:
            s.notes_slide.notes_text_frame.text = "\n".join(notes).strip()
        num = s.shapes.add_textbox(W - Cm(2.5), H - Cm(1.1), Cm(2), Cm(0.8))
        num.text_frame.paragraphs[0].text = str(n + 1); num.text_frame.paragraphs[0].font.size = Pt(10)
    prs.save(dst)
    print("wrote", dst, len(chunks), "slides")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    d = sys.argv[1] if len(sys.argv) > 1 else "docs"
    if os.path.exists(f"{d}/report.md"):
        md_to_pdf(f"{d}/report.md", f"{d}/report.pdf")
    if os.path.exists(f"{d}/slides.md"):
        md_to_pptx(f"{d}/slides.md", f"{d}/slides.pptx")
