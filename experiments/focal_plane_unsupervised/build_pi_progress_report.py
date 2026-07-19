"""Build the PI-facing progress report DOCX from verified experiment outputs."""
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
OUT = RESULTS / "PI_Progress_Report_Continuous_Prior_FP5D_Clustering.docx"

NAVY = "1F4D78"
BLUE = "2E74B5"
MUTED = "5B6573"
LIGHT = "F2F4F7"
PALE_BLUE = "E8EEF5"
PALE_GREEN = "EAF3EA"
INK = RGBColor(31, 77, 120)


def set_run_font(run, size=11, bold=False, color=None, name="Calibri"):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:ascii"), name)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), name)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd"); tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc = cell._tc; tc_pr = tc.get_or_add_tcPr()
    mar = tc_pr.first_child_found_in("w:tcMar")
    if mar is None:
        mar = OxmlElement("w:tcMar"); tc_pr.append(mar)
    for side, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = mar.find(qn(f"w:{side}"))
        if node is None:
            node = OxmlElement(f"w:{side}"); mar.append(node)
        node.set(qn("w:w"), str(value)); node.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths):
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.autofit = False
    table_pr = table._tbl.tblPr
    layout = table_pr.first_child_found_in("w:tblLayout")
    if layout is None:
        layout = OxmlElement("w:tblLayout"); table_pr.append(layout)
    layout.set(qn("w:type"), "fixed")
    tbl_w = table_pr.first_child_found_in("w:tblW")
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW"); table_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths))); tbl_w.set(qn("w:type"), "dxa")
    ind = table_pr.first_child_found_in("w:tblInd")
    if ind is None:
        ind = OxmlElement("w:tblInd"); table_pr.append(ind)
    ind.set(qn("w:w"), "120"); ind.set(qn("w:type"), "dxa")
    grid = table._tbl.tblGrid
    for grid_col, width in zip(grid.gridCol_lst, widths): grid_col.set(qn("w:w"), str(width))
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            cell.width = Inches(width / 1440)
            tc_w = cell._tc.get_or_add_tcPr().find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW"); cell._tc.get_or_add_tcPr().append(tc_w)
            tc_w.set(qn("w:w"), str(width)); tc_w.set(qn("w:type"), "dxa")
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell)


def set_repeat_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader"); header.set(qn("w:val"), "true"); tr_pr.append(header)


def write_cell(cell, text, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT, color=None, size=9.5):
    p = cell.paragraphs[0]; p.alignment = align
    p.paragraph_format.space_before = Pt(0); p.paragraph_format.space_after = Pt(0)
    r = p.add_run(str(text)); set_run_font(r, size=size, bold=bold, color=color)


def add_table(doc, headers, rows, widths):
    table = doc.add_table(rows=1, cols=len(headers))
    set_table_geometry(table, widths)
    set_repeat_header(table.rows[0])
    for c, text in zip(table.rows[0].cells, headers):
        shade(c, LIGHT); write_cell(c, text, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, color=NAVY)
    for row in rows:
        cells = table.add_row().cells
        for i, (c, value) in enumerate(zip(cells, row)):
            write_cell(c, value, align=WD_ALIGN_PARAGRAPH.CENTER if i > 0 else WD_ALIGN_PARAGRAPH.LEFT)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return table


def add_para(doc, text="", bold=False, italic=False, color=None, size=11, style=None, align=WD_ALIGN_PARAGRAPH.LEFT, before=0, after=6):
    p = doc.add_paragraph(style=style) if style else doc.add_paragraph()
    p.alignment = align; p.paragraph_format.space_before = Pt(before); p.paragraph_format.space_after = Pt(after); p.paragraph_format.line_spacing = 1.10
    r = p.add_run(text); set_run_font(r, size=size, bold=bold, color=color); r.italic = italic
    return p


def add_formula(doc, text):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4); p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text); set_run_font(r, size=11, name="Cambria Math", color=NAVY)
    return p


def add_caption(doc, text):
    p = add_para(doc, text, italic=True, color=MUTED, size=9, align=WD_ALIGN_PARAGRAPH.CENTER, before=2, after=10)
    return p


def add_picture(doc, path, width):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2); p.paragraph_format.space_after = Pt(0)
    p.add_run().add_picture(str(path), width=Inches(width))
    return p


def add_page_number(paragraph):
    run = paragraph.add_run("Page ")
    set_run_font(run, size=8.5, color=MUTED)
    fld = OxmlElement("w:fldSimple"); fld.set(qn("w:instr"), "PAGE")
    paragraph._p.append(fld)


def configure(doc):
    section = doc.sections[0]
    section.page_width = Inches(8.5); section.page_height = Inches(11)
    section.top_margin = section.bottom_margin = Inches(1)
    section.left_margin = section.right_margin = Inches(1)
    section.header_distance = Inches(.492); section.footer_distance = Inches(.492)
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"; normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.font.size = Pt(11); normal.paragraph_format.space_after = Pt(6); normal.paragraph_format.line_spacing = 1.10
    for name, size, color, before, after in (("Heading 1",16,BLUE,16,8),("Heading 2",13,BLUE,12,6),("Heading 3",12,NAVY,8,4)):
        s=styles[name]; s.font.name="Calibri"; s._element.rPr.rFonts.set(qn("w:eastAsia"),"Microsoft YaHei"); s.font.size=Pt(size); s.font.color.rgb=RGBColor.from_string(color); s.font.bold=True
        s.paragraph_format.space_before=Pt(before); s.paragraph_format.space_after=Pt(after); s.paragraph_format.line_spacing=1.10
    header=section.header.paragraphs[0]; header.alignment=WD_ALIGN_PARAGRAPH.LEFT
    r=header.add_run("SHMS optics calibration | FP 5D clustering progress"); set_run_font(r, size=8.5, color=MUTED)
    footer=section.footer.paragraphs[0]; footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    r=footer.add_run("Internal progress report | "); set_run_font(r,size=8.5,color=MUTED); add_page_number(footer)


def add_callout(doc, title, body):
    t = doc.add_table(rows=1, cols=1); set_table_geometry(t,[9360]); c=t.cell(0,0); shade(c,PALE_GREEN)
    p=c.paragraphs[0]; p.paragraph_format.space_before=Pt(2); p.paragraph_format.space_after=Pt(2)
    r=p.add_run(title+" "); set_run_font(r,size=10.5,bold=True,color=NAVY)
    r=p.add_run(body); set_run_font(r,size=10.5)
    doc.add_paragraph().paragraph_format.space_after=Pt(2)


def main():
    doc=Document(); configure(doc)
    # Memo masthead opening.
    add_para(doc,"RESEARCH PROGRESS REPORT",bold=True,color=BLUE,size=10,after=3)
    add_para(doc,"基于连续光学弱先验的可逆 FP 5D 聚类",bold=True,color=NAVY,size=24,after=4)
    add_para(doc,"将 focal-plane 事件云变换到与当前 sieve-plane 几何一致的完整五维聚类空间",color=MUTED,size=13,after=14)
    meta=[("To","PI"),("From","SHMS calibration / ML study"),("Date","2026-07-14"),("Dataset","Run 25521; current stage-2 sieve-plane reference"),("Status","Promising same-run complete-hole holdout result; cross-run validation pending")]
    for k,v in meta:
        p=doc.add_paragraph(); p.paragraph_format.space_after=Pt(2)
        r=p.add_run(k+": "); set_run_font(r,size=10.5,bold=True,color=NAVY)
        r=p.add_run(v); set_run_font(r,size=10.5)
    add_callout(doc,"Key finding.","Using only continuous reconstructed sieve_x, sieve_y, and P_gtr_y as weak training targets - not hole IDs, cluster centers, or foil labels - an invertible 5D transport yields HDBSCAN AMI 0.952 and ARI 0.916 on 44 completely held-out reference holes.")

    doc.add_heading("1. Executive summary", level=1)
    add_para(doc,"Direct clustering in raw focal-plane coordinates fails because the event clouds from different foils and sieve holes are thick, curved, and locally overlapping. The key observation is that the corresponding reference-hole centers form three much thinner, approximately two-dimensional sheets in the original five measured FP coordinates. The reported method uses current optics reconstruction only as a continuous weak geometric prior to learn a reversible reparameterization of this space.")
    add_table(doc,["Question","Result"],[
        ["Can raw FP 5D directly recover sieve holes?","No. Raw HDBSCAN either collapses into a few large groups or yields high noise."],
        ["Does raster conditioning alone solve the problem?","No. It stabilizes representation but does not separate holes by itself."],
        ["Can continuous optics targets flatten the geometry without discrete hole labels?","Yes, on the held-out-hole test split: 47 HDBSCAN clusters for 44 reference holes, 4.6% noise, AMI 0.952, ARI 0.916."],
        ["What remains unproven?","Generalization to an entirely unseen run and independence from the current optics reconstruction."],
    ],[2400,6960])

    doc.add_heading("2. Data source and evaluation protocol", level=1)
    add_para(doc,"Primary data source: stage-2 run 25521 table `stage2_25521_labeled_mech25x16p4_tol3_hdbscan_centerout_pen035.csv`. The reference sieve-plane result contains 220 `(foil, hole)` clusters (66 / 77 / 77 by foil) and is used only to construct the strict held-out-hole split and for final evaluation.")
    add_table(doc,["Role","Fields / handling"],[
        ["Measured FP input","P_dc_x_fp, P_dc_y_fp, P_dc_xp_fp, P_dc_yp_fp, P_rb_raster_frybRawAdc"],
        ["Continuous weak targets","sieve_x, sieve_y, P_gtr_y; per-event outputs of current optics reconstruction"],
        ["Explicitly excluded from fitting","cluster, foil_position, hole_id, hole_row, hole_col, cluster_center_x/y, foil_ytar_center"],
        ["Validation split","60,000 sampled events; 47,467 train events; 12,533 test events from 44 complete reference holes excluded before all fitting"],
    ],[2500,6860])
    add_para(doc,"The split uses reference cluster identity only to ensure that no events from a held-out hole appear in training. Those identities are never provided as model inputs, targets, or HDBSCAN constraints.",italic=True,color=MUTED,size=9.5)

    doc.add_heading("3. Geometric evidence in the measured FP 5D space", level=1)
    add_para(doc,"Each displayed point below is the event-average of one existing `(foil, hole)` reference cluster, expressed directly in measured FP coordinates. No dimensionality reduction is used in this plot. Grey links connect neighboring sieve-grid positions within each foil.")
    add_picture(doc,RESULTS/"12_fp5d_reference_center_coordinate_flows.png",6.35)
    add_caption(doc,"Figure 1. Direct coordinate slices of reference-hole centers. Color follows continuous sieve coordinate; each foil traces a curved center sheet rather than disconnected point islands.")
    add_table(doc,["Foil","Reference centers","First two center-PC variance","Effective center dimension","FP 5D vs sieve distance (Spearman)","Nearest 5D center is adjacent grid hole"],[
        ["0","66","0.748","2.903","0.607","0.833"],
        ["1","77","0.814","2.685","0.589","0.740"],
        ["2","77","0.801","2.357","0.530","0.649"],
    ],[600,850,1450,1250,2200,3010])
    add_para(doc,"Interpretation: center-level topology is locally meaningful, but event-level cloud thickness and cross-foil proximity obscure it. The median robust-scaled 5D distance between the same sieve-grid position across foil pairs is only 0.33-0.46.")

    doc.add_heading("4. Why raw 5D distance is poorly matched to the task", level=1)
    add_para(doc,"`fr_ybpm` is predominantly a raster-condition variable, not a hole-center coordinate. A between-center versus within-hole variance decomposition after robust scaling shows that it adds almost entirely within-hole spread, whereas `ypfp` carries the strongest center separation signal.")
    add_table(doc,["Variable","Center-between variance fraction","Within-hole variance fraction","Center signal / spread"],[
        ["xfp","0.110","0.894","0.123"],["yfp","0.405","0.604","0.671"],["xpfp","0.340","0.663","0.512"],["ypfp","0.853","0.231","3.689"],["fr_ybpm","0.003","0.997","0.003"],
    ],[1800,2500,2500,2560])
    add_para(doc,"Consequently, treating all five variables as equal Euclidean directions makes neighborhoods follow raster-induced thickness instead of the thinner optical center sheets.")

    doc.add_heading("5. Proposed method: continuous-prior invertible 5D transport", level=1)
    add_picture(doc,RESULTS/"15_continuous_prior_flow_schematic.png",6.35)
    add_caption(doc,"Figure 2. Training and clustering pipeline. Weak targets are used only during fitting; held-out reference-hole identities are reserved for final measurement.")
    doc.add_heading("5.1 Raster conditioning while retaining five dimensions", level=2)
    add_para(doc,"A cubic-spline Ridge model is fitted on training events only to predict the four optical FP coordinates from `fr_ybpm`. The transformed input is the residual optical vector plus the original raster coordinate:")
    add_formula(doc,"x′ = ( o - ô(fr_ybpm), fr_ybpm ) ∈ R⁵,   where o = (xfp, yfp, xpfp, ypfp).")
    add_para(doc,"This step does not discard the raster variable. It removes its smooth broadening effect from the optical coordinates while retaining it as a fifth residual/context direction.")
    doc.add_heading("5.2 Invertible weakly supervised transport", level=2)
    add_para(doc,"A six-coupling-layer RealNVP learns an exactly invertible map F: R⁵ → R⁵. Its first three output coordinates are softly aligned to standardized continuous optics targets, while the final two retain non-target residual information.")
    add_formula(doc,"z = F(x′) = (z₁,z₂,z₃,z₄,z₅),   (z₁,z₂,z₃) ≈ standardize(sieve_x, sieve_y, P_gtr_y).")
    add_formula(doc,"L = MSE((z₁,z₂,z₃), ỹ) + 0.02 × MSE((z₄,z₅), (x′₄,x′₅)).")
    add_para(doc,"The numerical forward-inverse round-trip maximum absolute error is 6.84×10⁻⁵. The model therefore rearranges five-dimensional geometry rather than creating a lower-dimensional embedding.")
    doc.add_heading("5.3 Final clustering distance", level=2)
    add_formula(doc,"d²(i,j) = Σₖ₌₁³ (zᵢₖ - zⱼₖ)² + 0.10 × Σₖ₌₄⁵ (zᵢₖ - zⱼₖ)².")
    add_para(doc,"The last two residual dimensions retain nonzero weight. HDBSCAN is then run without prescribing the number of clusters.")

    doc.add_heading("6. Held-out-hole performance", level=1)
    add_picture(doc,RESULTS/"14_pi_performance_comparison.png",6.35)
    add_caption(doc,"Figure 3. Comparable HDBSCAN results on the same 44-hole held-out split. The gold bar uses discrete cluster centers and is shown only as a stronger-supervision reference, not as the proposed method.")
    add_table(doc,["Space / supervision","Clusters","Noise","AMI","ARI","Comment"],[
        ["Raw 5D HDBSCAN","128","71.0%","0.230","-0.002","High-noise fragmented baseline"],
        ["Raster-conditioned raw 5D","128","71.0%","0.230","-0.002","Conditioning alone is insufficient"],
        ["Continuous-prior flow, equal 5D","46","5.1%","0.947","0.904","All five output dimensions retained"],
        ["Continuous-prior flow, residual weight 0.10","47","4.6%","0.952","0.916","Selected proposed configuration"],
        ["Discrete-center flow reference","46","3.3%","0.958","0.928","Stronger supervision; not the proposed configuration"],
    ],[2700,800,800,800,800,3460])
    add_callout(doc,"Decision-relevant result.","The continuous-prior configuration nearly matches the stronger discrete-center reference while explicitly excluding cluster centers and all discrete hole information from fitting.")

    doc.add_heading("7. Reconstruction-space closure test", level=1)
    add_para(doc,"The cluster labels are inferred in the transformed measured FP 5D space. For a physics-facing closure check, we project the test-event labels back onto the independently reconstructed sieve coordinates and reconstructed target coordinate used only as continuous weak targets. This is not an additional fit and does not use reference hole identities to assign colors.")
    add_picture(doc,RESULTS/"16_continuous_prior_flow_backprojection_sieve_ytar.png",6.35)
    add_caption(doc,"Figure 4. Held-out-hole flow-HDBSCAN labels projected back to current reconstruction. Top: reconstructed sieve plane; bottom: reconstructed ytar versus sieve-y. Each color is an inferred cluster; grey points are the 4.6% HDBSCAN noise. The compact islands and foil-dependent ytar bands demonstrate reconstruction-space closure, while not constituting independent truth validation.")

    doc.add_heading("8. Interpretation and limitations", level=1)
    add_para(doc,"The result supports a specific conclusion: raw FP measurements contain the information needed to recover the current sieve-plane topology, but raw Euclidean geometry does not expose it. Current continuous optics reconstruction provides a useful direction field for unfolding the geometry. The method should therefore be viewed as a focal-plane clustering and quality-control layer anchored by weak continuous physics, not as a replacement for independent calibration truth.")
    add_table(doc,["What is established","What is not yet established"],[
        ["Complete held-out holes can be assigned consistently after continuous-prior 5D transport.","Generalization to entirely unseen runs."],
        ["No discrete cluster centers, hole IDs, foil labels, or mechanical grid coordinates enter fitting.","Independence from the current optics reconstruction, because sieve_x/y and P_gtr_y remain reconstruction outputs."],
        ["The map is numerically invertible and keeps all five dimensions.","Stability under alternative target calibration, detector conditions, and run-to-run raster behavior."],
    ],[4680,4680])

    doc.add_heading("9. Recommended next experiment", level=1)
    add_para(doc,"Run-level generalization is now the critical gate. Select one or more runs that are entirely absent from training, scaler fitting, raster conditioning, and HDBSCAN parameter selection. Train only on the remaining runs, apply the frozen transport to the held-out run, and compare against that run's sieve-plane reference once at the end.")
    # Keep the four-step protocol intact: splitting a table across pages makes
    # the first column unreadable in Word's pagination.
    doc.add_page_break()
    add_table(doc,["Step","Required control","Success measure"],[
        ["1. Freeze split","No held-out-run event enters any fit or hyperparameter scan","No data leakage"],
        ["2. Train map","Use only FP 5D and continuous sieve_x/y, P_gtr_y from training runs","Stable round-trip and target error"],
        ["3. Cluster held-out","Apply frozen distance and fixed HDBSCAN configuration","Noise <10%, sensible cluster count"],
        ["4. Evaluate once","Compare only after clustering","AMI/ARI materially above raw FP baseline"],
    ],[1600,3800,3960])

    doc.add_heading("Appendix A. Data and code provenance", level=1)
    add_para(doc,"Data: `SHMS_Calibration_NN/dataset/stage2_25521_labeled_mech25x16p4_tol3_hdbscan_centerout_pen035.csv`.")
    add_para(doc,"Primary experiment: `experiments/focal_plane_unsupervised/run_continuous_prior_flow_metric.py`.")
    add_para(doc,"Center-flow analysis: `experiments/focal_plane_unsupervised/run_fp5d_center_flow_study.py`.")
    add_para(doc,"Machine-readable results: `results/continuous_prior_flow_metric_holeholdout_summary.json` and `results/continuous_prior_flow_hdbscan_holeholdout_scan.csv`.")
    add_para(doc,"All values in this report are derived from the above local data and experiment outputs; no external datasets or literature-derived performance numbers are used.",italic=True,color=MUTED,size=9.5)

    doc.core_properties.title = "PI Progress Report - Continuous Prior FP5D Clustering"
    doc.core_properties.subject = "SHMS optics calibration and focal-plane clustering"
    doc.core_properties.author = "Codex"
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__": main()
