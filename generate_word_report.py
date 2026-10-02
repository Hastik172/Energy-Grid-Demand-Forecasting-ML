"""
Generates the comprehensive academic research Word Document (.docx)
for EnergyGrid-AI project according to the reference specification.
"""

import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def style_table(table, header_bg="1B365D", alt_bg="F7FAFC", col_widths=None):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(table.rows):
        # Prevent row splitting across pages
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        if i == 0:
            trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
            
        for j, cell in enumerate(row.cells):
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=120, bottom=120, left=140, right=140)
            if col_widths and j < len(col_widths):
                cell.width = Inches(col_widths[j])
            if i == 0:
                set_cell_background(cell, header_bg)
                for paragraph in cell.paragraphs:
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in paragraph.runs:
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(255, 255, 255)
                        run.font.size = Pt(9.5)
                        run.font.name = 'Calibri'
            else:
                if i % 2 == 1:
                    set_cell_background(cell, "FFFFFF")
                else:
                    set_cell_background(cell, alt_bg)
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(9.0)
                        run.font.name = 'Calibri'
                        run.font.color.rgb = RGBColor(30, 41, 59)

def create_report():
    doc = Document()
    
    # Page setup
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        section.different_first_page_header_footer = True
        
        # Header / Footer
        footer = section.footer
        f_p = footer.paragraphs[0]
        f_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        f_run = f_p.add_run("EnergyGrid-AI — Unit 1 Research Report | B.Tech CSE")
        f_run.font.size = Pt(8.5)
        f_run.font.color.rgb = RGBColor(148, 163, 184)
        
        header = section.header
        h_p = header.paragraphs[0]
        h_p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        h_run = h_p.add_run("ENERGYGRID-AI: Probabilistic Demand & Renewable Forecasting")
        h_run.font.size = Pt(8.5)
        h_run.font.color.rgb = RGBColor(148, 163, 184)

    # -------------------------------------------------------------
    # PAGE 1: TITLE PAGE
    # -------------------------------------------------------------
    p_title_space = doc.add_paragraph()
    p_title_space.paragraph_format.space_before = Pt(72)
    
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run("ENERGYGRID-AI")
    r_title.font.name = 'Calibri'
    r_title.font.size = Pt(32)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(15, 23, 42) # Dark Slate
    p_title.paragraph_format.space_after = Pt(12)
    
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = p_sub.add_run("Probabilistic Early-Demand & Renewable Power Forecasting\nwith Uncertainty Quantification")
    r_sub.font.name = 'Calibri'
    r_sub.font.size = Pt(16)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(30, 58, 138) # Navy Blue
    p_sub.paragraph_format.space_after = Pt(36)
    
    p_deg = doc.add_paragraph()
    p_deg.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_deg = p_deg.add_run("B.Tech Computer Science and Engineering\nMachine Learning — Unit 1 Research Project Report")
    r_deg.font.name = 'Calibri'
    r_deg.font.size = Pt(13)
    r_deg.font.color.rgb = RGBColor(71, 85, 105)
    p_deg.paragraph_format.space_after = Pt(64)
    
    p_by = doc.add_paragraph()
    p_by.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_by = p_by.add_run("By:\n")
    r_by.font.name = 'Calibri'
    r_by.font.size = Pt(12)
    r_by.font.bold = True
    r_by.font.color.rgb = RGBColor(15, 23, 42)
    
    members = [
        "Vaishnavi Jagtap [RA2411003011544]",
        "Suhani Singh [RA2411003011538]",
        "Ali Ahmed Flexwala [RA2411003011560]",
        "Hastik Pangi [RA2411003011561]"
    ]
    for m in members:
        p_mem = doc.add_paragraph()
        p_mem.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_mem = p_mem.add_run(m)
        r_mem.font.name = 'Calibri'
        r_mem.font.size = Pt(12)
        r_mem.font.color.rgb = RGBColor(30, 41, 59)
        p_mem.paragraph_format.space_after = Pt(4)
        
    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 2: TABLE OF CONTENTS
    # -------------------------------------------------------------
    p_toc_head = doc.add_paragraph()
    r_toc_head = p_toc_head.add_run("Table of Contents")
    r_toc_head.font.name = 'Calibri'
    r_toc_head.font.size = Pt(20)
    r_toc_head.font.bold = True
    r_toc_head.font.color.rgb = RGBColor(15, 23, 42)
    p_toc_head.paragraph_format.space_after = Pt(18)
    
    toc_items = [
        "1. Introduction",
        "2. Dataset and Data Preparation",
        "3. Statistical and Probabilistic Methodology",
        "4. Machine Learning Models",
        "5. Polynomial Response Modelling and Curve Fitting",
        "6. System Architecture",
        "7. Implementation",
        "8. Experiments and Results",
        "9. Dashboard",
        "10. Testing and Reproducibility",
        "11. Applications",
        "12. Limitations",
        "13. Future Scope",
        "14. Conclusion",
        "15. References",
        "Appendix: Final Project Outputs & Experimental Results"
    ]
    for item in toc_items:
        p_item = doc.add_paragraph()
        r_item = p_item.add_run(item)
        r_item.font.name = 'Calibri'
        r_item.font.size = Pt(11)
        r_item.font.color.rgb = RGBColor(30, 41, 59)
        p_item.paragraph_format.space_after = Pt(6)
        
    doc.add_page_break()

    # -------------------------------------------------------------
    # PAGE 3: ABSTRACT & KEYWORDS
    # -------------------------------------------------------------
    p_abs_head = doc.add_paragraph()
    r_abs_head = p_abs_head.add_run("Abstract")
    r_abs_head.font.name = 'Calibri'
    r_abs_head.font.size = Pt(20)
    r_abs_head.font.bold = True
    r_abs_head.font.color.rgb = RGBColor(15, 23, 42)
    p_abs_head.paragraph_format.space_after = Pt(12)
    
    abstract_text = (
        "Electrical power grids form the backbone of industrial infrastructure, yet modern energy management faces "
        "growing challenges from weather volatility and the rapid deployment of intermittent renewable power sources "
        "(solar and wind). Accurate electrical demand forecasting and uncertainty quantification are crucial for system "
        "stability, unit commitment, and blackout prevention. Conventional heuristic or static schedule strategies either "
        "fail to anticipate extreme weather surges—incurring severe safety risks and emergency peaker dispatch costs—or "
        "over-commit spinning reserves, causing substantial economic and carbon penalties.\n\n"
        "This project presents EnergyGrid-AI, an end-to-end, statistically grounded machine learning and uncertainty "
        "quantification system built strictly upon Unit 1 Machine Learning and Probability Theory foundations. The system "
        "evaluates 8,760 hourly records spanning a complete calendar year (2023) with 8 primary features (electricity demand, "
        "solar generation, wind generation, temperature, relative humidity, wind speed, cloud cover, and timestamps). A rigorous "
        "data engineering pipeline resolves physical sensor transmission defects by removing 5 duplicate logs and imputing "
        "25 missing records through time-aware linear interpolation.\n\n"
        "EnergyGrid-AI seamlessly unites four foundational Unit 1 pillars into a coherent computational architecture. "
        "First, descriptive statistics, quantiles, and probability density functions (parametric Gaussian vs. non-parametric "
        "Kernel Density Estimation) characterize demand distributions and establish vital operational thresholds "
        "(Q05 = 16,599 MW, Median = 25,396 MW, Q95 = 31,322 MW), while Kolmogorov's addition theorem is verified empirically "
        "with an exact 0.0 discrepancy. Second, Bayesian inference evaluates posterior high-demand risk given extreme weather "
        "observations, demonstrating a 1.79× belief update when ambient temperature exceeds the 80th percentile. Third, a "
        "supervised multivariate regression model with domain-engineered polynomial features (thermodynamic temperature quadratic "
        "terms and harmonic cyclic diurnal encodings) achieves an R² of 0.8092 and MAE of 1,332.72 MW on a strict chronological "
        "train/test split (7,008 train / 1,752 test hours), outperforming raw linear regression by over 64%. Fourth, unsupervised "
        "K-Means clustering (selected K = 4 via elbow analysis and silhouette metrics) discovers four distinct operational regimes "
        "(Winter Heating Peak, Baseload Night, Summer Cooling Peak, and Windy Transition) with zero manual labeling. In addition, "
        "Uncertainty Quantification (UQ) constructs calibrated 90% empirical prediction intervals that achieve an 89.95% empirical "
        "coverage on unseen test data.\n\n"
        "The complete system is accessible through both an interactive terminal interface and an 8-page full-stack web dashboard "
        "with a live Bayesian Risk Engine, and is verified through an automated reproducibility test suite. This report documents "
        "the theoretical formulations, implementation, experimental outcomes, dashboard architecture, testing integrity, "
        "and ethical boundaries of EnergyGrid-AI."
    )
    p_abs = doc.add_paragraph()
    r_abs = p_abs.add_run(abstract_text)
    r_abs.font.name = 'Calibri'
    r_abs.font.size = Pt(10.5)
    r_abs.font.color.rgb = RGBColor(30, 41, 59)
    p_abs.paragraph_format.line_spacing = 1.15
    p_abs.paragraph_format.space_after = Pt(16)
    
    p_kw = doc.add_paragraph()
    r_kwh = p_kw.add_run("Keywords: ")
    r_kwh.font.name = 'Calibri'
    r_kwh.font.size = Pt(10)
    r_kwh.font.bold = True
    r_kwh.font.color.rgb = RGBColor(15, 23, 42)
    r_kw = p_kw.add_run("Machine Learning, Power Systems, Energy Demand Forecasting, Renewable Power, Probability Theory, Kolmogorov Axioms, Bayes' Rule, Kernel Density Estimation, Polynomial Curve Fitting, Supervised Regression, K-Means Clustering, Uncertainty Quantification, Prediction Intervals, Web Dashboard.")
    r_kw.font.name = 'Calibri'
    r_kw.font.size = Pt(10)
    r_kw.font.color.rgb = RGBColor(71, 85, 105)
    
    doc.add_page_break()

    # Helper function for adding sections
    def add_heading_1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(18)
        p.paragraph_format.space_after = Pt(8)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        r.font.name = 'Calibri'
        r.font.size = Pt(16)
        r.font.bold = True
        r.font.color.rgb = RGBColor(27, 54, 93) # Dark Navy
        return p

    def add_heading_2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        r.font.name = 'Calibri'
        r.font.size = Pt(13)
        r.font.bold = True
        r.font.color.rgb = RGBColor(44, 82, 130) # Slate Blue
        return p

    def add_body(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.15
        r = p.add_run(text)
        r.font.name = 'Calibri'
        r.font.size = Pt(10.5)
        r.font.color.rgb = RGBColor(30, 41, 59)
        return p

    def add_bullet(bold_prefix, text):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        r_pre = p.add_run(bold_prefix + ": ")
        r_pre.font.name = 'Calibri'
        r_pre.font.size = Pt(10.5)
        r_pre.font.bold = True
        r_pre.font.color.rgb = RGBColor(15, 23, 42)
        r_txt = p.add_run(text)
        r_txt.font.name = 'Calibri'
        r_txt.font.size = Pt(10.5)
        r_txt.font.color.rgb = RGBColor(30, 41, 59)
        return p

    # -------------------------------------------------------------
    # 1. INTRODUCTION
    # -------------------------------------------------------------
    add_heading_1("1. Introduction")
    
    add_heading_2("1.1 Background")
    add_body(
        "Electrical energy powers every facet of modern society—from residential life to heavy industrial plants, "
        "transportation networks, and computational data centers. Unlike conventional supply chains where goods can be stored "
        "cost-effectively in inventory, alternating current (AC) power grids require instantaneous, second-by-second balance "
        "between total generation and total consumer demand. Any persistent discrepancy induces system frequency deviations, "
        "which if left unmitigated, triggers protective generation disconnects and catastrophic cascading outages."
    )
    add_body(
        "EnergyGrid-AI demonstrates, on real continuous grid time series, how the mathematical principles of Unit 1 Machine "
        "Learning—spanning continuous random variables, probability densities, Kolmogorov axioms, Bayesian inference, "
        "supervised multivariate regression, unsupervised clustering, and prediction interval uncertainty quantification—can "
        "be formulated into a single, cohesive, production-grade intelligence pipeline for electric utility operations."
    )

    add_heading_2("1.2 Problem Statement")
    add_body(
        "Electrical demand response is notoriously non-linear. Thermodynamic heat loss and air-conditioning refrigeration "
        "induce a distinctive parabolic U-curve relationship with ambient temperature. Simultaneously, intermittent renewable "
        "generation (solar and wind) introduces severe volatility into net load. Traditional engineering models based on static "
        "heuristics or simple linear regression fail to capture this curvature, yielding negative generalization and poor reliability. "
        "A rigorous monitoring system must therefore: (a) quantify statistical distributions and percentiles across energy variables, "
        "(b) evaluate conditional operational risks using formal Bayesian updating, (c) formulate feature-engineered regression models "
        "that capture thermodynamic physics, (d) discover operational grid states without manual labeling, and (e) construct "
        "calibrated prediction intervals that provide grid operators with actionable confidence margins."
    )

    add_heading_2("1.3 Motivation")
    add_bullet("Verifiable Statistical Grounding", "Deriving all means, variances, quantiles, and probability density functions directly from empirical grid data rather than arbitrary assumptions.")
    add_bullet("Probabilistic Risk Updating", "Using Bayes' Theorem to convert baseline prior probabilities into quantified posterior risks when extreme weather alerts are detected.")
    add_bullet("Bridging Supervised & Unsupervised Learning", "Combining supervised polynomial regression for continuous MW demand prediction with unsupervised K-Means clustering for automatic grid regime classification.")
    add_bullet("Actionable Uncertainty Quantification", "Equipping dispatchers with empirical 90% and 95% prediction intervals to replace unsafe point forecasts.")

    add_heading_2("1.4 Objectives")
    add_bullet("Objective 1", "Formulate an adaptive data cleaning and time-aware linear interpolation pipeline to resolve sensor packet drops and duplicate transmissions.")
    add_bullet("Objective 2", "Empirically verify Kolmogorov's probability axioms and addition rules on high-demand and high-renewable energy events.")
    add_bullet("Objective 3", "Construct a 3-state discrete random variable for demand and calculate its PMF, CDF, expectation, and variance.")
    add_bullet("Objective 4", "Formulate Bayes' Rule to evaluate the posterior risk of peak load conditioned on extreme heatwaves.")
    add_bullet("Objective 5", "Quantify feature co-movements and examine statistical independence using covariance and Pearson correlation.")
    add_bullet("Objective 6", "Perform polynomial curve fitting (Degrees 1 through 4) on temperature-demand dynamics to demonstrate the PRML bias-variance tradeoff.")
    add_bullet("Objective 7", "Train and evaluate a multivariate supervised regression model on a strict 80/20 chronological split, comparing raw features against engineered polynomial features.")
    add_bullet("Objective 8", "Apply unsupervised K-Means clustering (K=4) to discover operational dispatch regimes without labels.")
    add_bullet("Objective 9", "Formulate empirical quantile and Gaussian prediction intervals and evaluate empirical coverage probabilities.")
    add_bullet("Objective 10", "Deploy the end-to-end system through an interactive terminal interface and an 8-page full-stack web dashboard.")

    add_heading_2("1.5 Scope")
    add_body(
        "The scope of EnergyGrid-AI strictly encompasses the statistical, probabilistic, and machine learning foundations defined "
        "in Unit 1. Algorithmic methods are restricted to ordinary least squares, polynomial basis expansions, classical probability, "
        "and Lloyd's centroid clustering. Complex deep neural architectures (LSTM, GRU, Transformers) and autoregressive models (ARIMA) "
        "are explicitly outside the current implementation scope and reserved for future units."
    )

    # -------------------------------------------------------------
    # 2. DATASET AND DATA PREPARATION
    # -------------------------------------------------------------
    add_heading_1("2. Dataset and Data Preparation")
    
    add_heading_2("2.1 Dataset Source and Composition")
    add_body(
        "EnergyGrid-AI is built on an annual hourly grid dataset modeling physical power system dynamics across 8,760 chronological "
        "hours (January 1, 2023 to December 31, 2023). The dataset encompasses 8 primary features:"
    )
    
    # Table 2.1
    t21 = doc.add_table(rows=9, cols=5)
    headers21 = ["Variable", "Symbol", "Unit", "Physical Range", "Description"]
    for j, h in enumerate(headers21):
        t21.cell(0, j).paragraphs[0].text = h
    
    data21 = [
        ["Timestamp", "t", "YYYY-MM-DD HH:MM", "2023-01-01 to 2023-12-31", "Chronological hourly index"],
        ["Electricity Demand", "Dt", "Megawatts (MW)", "12,330 – 40,031 MW", "Total electrical grid load (Target)"],
        ["Solar Generation", "St", "Megawatts (MW)", "0 – 7,845 MW", "Solar PV farm output (diurnal)"],
        ["Wind Generation", "Wt", "Megawatts (MW)", "0 – 10,000 MW", "Wind turbine farm output (Weibull)"],
        ["Ambient Temperature", "Tt", "Celsius (°C)", "-7.56 – 39.15 °C", "Meteorological station surface temp"],
        ["Relative Humidity", "Ht", "Percentage (%)", "15.0 – 99.0 %", "Atmospheric water saturation"],
        ["Wind Speed", "vt", "Meters/sec (m/s)", "0.5 – 26.0 m/s", "Anemometer speed at hub height (80m)"],
        ["Cloud Cover", "Ct", "Percentage (%)", "0.0 – 100.0 %", "Sky cloud fraction (Beer-Lambert)"]
    ]
    for i, row in enumerate(data21):
        for j, val in enumerate(row):
            t21.cell(i+1, j).paragraphs[0].text = val
    style_table(t21, col_widths=[1.5, 0.6, 1.2, 1.4, 2.0])
    p_cap21 = doc.add_paragraph()
    p_cap21.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_cap21 = p_cap21.add_run("Table 2.1 — Dataset feature attributes and physical boundaries")
    r_cap21.font.size = Pt(8.5)
    r_cap21.font.italic = True
    p_cap21.paragraph_format.space_after = Pt(12)

    add_heading_2("2.2 Time-Aware Imputation and Preprocessing")
    add_body(
        "Sensor network artifacts—namely duplicate packets and intermittent data dropouts—were identified and resolved:\n"
        "1. Duplicate Records: 5 duplicate timestamps were removed, restoring the canonical shape to 8,760 hours.\n"
        "2. Missing Values: 25 missing values distributed across columns were imputed using time-aware linear interpolation:\n"
        "   xt = xt0 + ((xt1 - xt0) / (t1 - t0)) * (t - t0)\n"
        "Linear interpolation preserves local physical continuity in time series, whereas naive mean imputation introduces unnatural step discontinuities."
    )

    add_heading_2("2.3 State of Demand Categories")
    add_body(
        "Continuous electrical demand is discretized into three operating condition classes based on empirical tertile quantiles "
        "(Q0.333 = 23,061.40 MW and Q0.667 = 26,866.90 MW):"
    )
    
    # Table 2.2
    t22 = doc.add_table(rows=4, cols=3)
    headers22 = ["Condition Class", "Demand Range (MW)", "Operational Interpretation"]
    for j, h in enumerate(headers22):
        t22.cell(0, j).paragraphs[0].text = h
    data22 = [
        ["Low Demand (State 0)", "< 23,061.40 MW", "Off-peak nocturnal baseload; low grid stress"],
        ["Normal Demand (State 1)", "23,061.40 – 26,866.90 MW", "Typical daytime residential and commercial activity"],
        ["High Demand (State 2)", "> 26,866.90 MW", "Peak grid stress; triggers spinning reserve and peaker dispatch"]
    ]
    for i, row in enumerate(data22):
        for j, val in enumerate(row):
            t22.cell(i+1, j).paragraphs[0].text = val
    style_table(t22, col_widths=[2.0, 2.0, 2.7])
    p_cap22 = doc.add_paragraph()
    p_cap22.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_cap22 = p_cap22.add_run("Table 2.2 — Discretized demand condition classes")
    r_cap22.font.size = Pt(8.5)
    r_cap22.font.italic = True
    p_cap22.paragraph_format.space_after = Pt(12)

    add_heading_2("2.4 Mapping of Unit 1 Concepts to EnergyGrid-AI")
    add_body("Every concept in the Unit 1 syllabus is directly manifested in the codebase:")
    
    # Table 2.3
    t23 = doc.add_table(rows=16, cols=2)
    t23.cell(0, 0).paragraphs[0].text = "Unit 1 Academic Concept"
    t23.cell(0, 1).paragraphs[0].text = "EnergyGrid-AI System Implementation"
    data23 = [
        ["What is Machine Learning", "End-to-end task, experience, and performance formulation"],
        ["Supervised Learning", "Multivariate regression predicting continuous MW demand"],
        ["Unsupervised Learning", "K-Means clustering identifying 4 grid operational regimes"],
        ["Polynomial Curve Fitting", "Thermodynamic U-curve temperature-demand modeling"],
        ["Probability Theory Axioms", "Empirical Kolmogorov verification with zero discrepancy"],
        ["Discrete Random Variables", "3-state demand PMF, CDF, expectation, and variance"],
        ["Bayes' Rule", "Posterior risk update conditioned on extreme heatwaves"],
        ["Independence & Correlation", "Pairwise association, Pearson r, and non-linear dependence"],
        ["Continuous Random Variables", "Continuous distributions of load, solar, wind, and temp"],
        ["Quantiles and IQR", "Operational percentiles (Q05, Q50, Q95) & Tukey anomaly bounds"],
        ["Probability Density Functions", "Gaussian parametric PDF vs. non-parametric Kernel Density (KDE)"],
        ["Expectation and Covariance", "Fleet-wide mean vectors and feature covariance matrices"],
        ["Train-Test Split", "Chronological 80/20 partitioning (7,008 train / 1,752 test hours)"],
        ["Model Evaluation", "MAE, MSE, RMSE, and R² regression performance evaluation"],
        ["Uncertainty Quantification", "Calibrated 90% empirical and 95% Gaussian prediction intervals"]
    ]
    for i, row in enumerate(data23):
        for j, val in enumerate(row):
            t23.cell(i+1, j).paragraphs[0].text = val
    style_table(t23, col_widths=[2.8, 3.9])
    p_cap23 = doc.add_paragraph()
    p_cap23.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_cap23 = p_cap23.add_run("Table 2.3 — Mapping of Unit 1 syllabus concepts to implementation modules")
    r_cap23.font.size = Pt(8.5)
    r_cap23.font.italic = True
    p_cap23.paragraph_format.space_after = Pt(12)

    # -------------------------------------------------------------
    # 3. STATISTICAL AND PROBABILISTIC METHODOLOGY
    # -------------------------------------------------------------
    add_heading_1("3. Statistical and Probabilistic Methodology")
    
    add_heading_2("3.1 Descriptive Statistics on Demand and Environmental Variables")
    add_body(
        "Sample statistics were computed using unbiased Bessel-corrected estimators across all 8,760 hourly observations:"
    )
    
    # Table 3.1
    t31 = doc.add_table(rows=6, cols=8)
    headers31 = ["Variable", "Mean (μ)", "Std Dev (σ)", "Median", "Min", "Max", "IQR", "95th Pct"]
    for j, h in enumerate(headers31):
        t31.cell(0, j).paragraphs[0].text = h
    data31 = [
        ["Demand (MW)", "24,611.16", "4,535.11", "25,395.92", "12,330.86", "40,031.69", "5,899.93", "31,322.15"],
        ["Temp (°C)", "15.98", "9.46", "15.78", "-7.56", "39.15", "15.32", "30.97"],
        ["Solar (MW)", "1,147.36", "1,525.89", "346.88", "0.00", "7,844.57", "1,884.19", "4,456.73"],
        ["Wind (MW)", "1,460.36", "2,470.59", "263.16", "0.00", "10,000.00", "1,676.68", "8,287.64"],
        ["Renewable (MW)", "2,607.72", "2,754.04", "1,747.84", "0.00", "16,195.75", "3,294.49", "9,284.89"]
    ]
    for i, row in enumerate(data31):
        for j, val in enumerate(row):
            t31.cell(i+1, j).paragraphs[0].text = val
    style_table(t31, col_widths=[1.5, 0.8, 0.8, 0.8, 0.7, 0.8, 0.7, 0.7])
    p_cap31 = doc.add_paragraph()
    p_cap31.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_cap31 = p_cap31.add_run("Table 3.1 — Descriptive statistical parameters across continuous grid variables")
    r_cap31.font.size = Pt(8.5)
    r_cap31.font.italic = True
    p_cap31.paragraph_format.space_after = Pt(12)

    # Insert Fig 1 & 2
    if os.path.exists("figures/01_demand_timeseries.png"):
        p_img1 = doc.add_paragraph()
        p_img1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img1.paragraph_format.space_before = Pt(8)
        p_img1.paragraph_format.space_after = Pt(4)
        doc.add_picture("figures/01_demand_timeseries.png", width=Inches(5.8))
        p_c1 = doc.add_paragraph()
        p_c1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_c1 = p_c1.add_run("Figure 3.1 — Annual electricity demand profile and 2-week diurnal cycle zoom")
        r_c1.font.size = Pt(8.5)
        r_c1.font.italic = True
        p_c1.paragraph_format.space_after = Pt(12)

    add_heading_2("3.2 Probability Density and Kolmogorov Axioms")
    add_body(
        "Kolmogorov's third probability axiom states that for any two events A and B, the general addition theorem holds:\n"
        "P(A ∪ B) = P(A) + P(B) - P(A ∩ B)\n\n"
        "Empirical verification on energy events:\n"
        "• Event A: High Demand (Dt > Q0.75 = 27,714.0 MW) → P(A) = 2,190 / 8,760 = 0.2500\n"
        "• Event B: High Renewable (Rt > Q0.75 = 3,760.2 MW) → P(B) = 2,190 / 8,760 = 0.2500\n"
        "• Joint Probability: P(A ∩ B) = 0.0626\n"
        "• Empirical Union: P(A ∪ B) = 0.4374\n"
        "• Axiomatic Formula: 0.2500 + 0.2500 - 0.0626 = 0.4374\n"
        "• Discrepancy: |0.4374 - 0.4374| = 0.000000 (Exact Verification)"
    )

    if os.path.exists("figures/04_probability_density.png"):
        doc.add_picture("figures/04_probability_density.png", width=Inches(5.8))
        p_c4 = doc.add_paragraph()
        p_c4.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_c4 = p_c4.add_run("Figure 3.2 — Gaussian parametric PDF vs. non-parametric Kernel Density Estimation (KDE)")
        r_c4.font.size = Pt(8.5)
        r_c4.font.italic = True
        p_c4.paragraph_format.space_after = Pt(12)

    add_heading_2("3.3 Bayesian Inference Applied to High-Demand Risk")
    add_body(
        "Bayes' Theorem formalizes how prior beliefs update upon observing new meteorological evidence:\n"
        "P(High Demand | High Temp) = [P(High Temp | High Demand) * P(High Demand)] / P(High Temp)"
    )
    
    # Table 3.2
    t32 = doc.add_table(rows=5, cols=2)
    t32.cell(0, 0).paragraphs[0].text = "Bayesian Component"
    t32.cell(0, 1).paragraphs[0].text = "Empirical Value"
    data32 = [
        ["Prior — P(High Demand)", "0.2500 (25.00%)"],
        ["Likelihood — P(High Temp | High Demand)", "0.3575 (35.75%)"],
        ["Evidence — P(High Temp)", "0.1997 (19.97%)"],
        ["Posterior — P(High Demand | High Temp)", "0.4477 (44.77%)"]
    ]
    for i, row in enumerate(data32):
        for j, val in enumerate(row):
            t32.cell(i+1, j).paragraphs[0].text = val
    style_table(t32, col_widths=[3.5, 3.2])
    p_cap32 = doc.add_paragraph()
    p_cap32.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_cap32 = p_cap32.add_run("Table 3.2 — Worked Bayesian posterior risk updating example")
    r_cap32.font.size = Pt(8.5)
    r_cap32.font.italic = True
    p_cap32.paragraph_format.space_after = Pt(12)

    add_body(
        "Concretely, observing an ambient temperature above 25.1°C (80th percentile) raises the probability of experiencing "
        "high grid demand from a 25.00% prior to a 44.77% posterior—representing a 1.79× risk update ratio. This provides grid "
        "dispatchers with clear quantitative evidence to activate fast-responding peaking turbines."
    )

    if os.path.exists("figures/08_bayes_probability.png"):
        doc.add_picture("figures/08_bayes_probability.png", width=Inches(5.8))
        p_c8 = doc.add_paragraph()
        p_c8.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_c8 = p_c8.add_run("Figure 3.3 — Prior vs. posterior risk updating via Bayes' Rule")
        r_c8.font.size = Pt(8.5)
        r_c8.font.italic = True
        p_c8.paragraph_format.space_after = Pt(12)

    add_heading_2("3.4 Covariance, Correlation, and Independence")
    add_body(
        "Covariance Cov(X,Y) and Pearson correlation r were computed across continuous features. "
        "The results reveal the non-linear correlation paradox:"
    )
    
    # Table 3.3
    t33 = doc.add_table(rows=4, cols=4)
    headers33 = ["Feature Pair", "Covariance", "Pearson r", "Physical / Statistical Interpretation"]
    for j, h in enumerate(headers33):
        t33.cell(0, j).paragraphs[0].text = h
    data33 = [
        ["Solar & Demand", "+2,355,624.74 MW²", "+0.3404", "Positive co-movement: solar peaks midday alongside commercial demand"],
        ["Wind & Demand", "-1,388,500.92 MW²", "-0.1239", "Weak negative correlation: night winds coincide with off-peak load"],
        ["Temperature & Demand", "-2,076.74 MW·°C", "-0.0484", "Linear cancellation of parabolic heating/cooling U-curve"]
    ]
    for i, row in enumerate(data33):
        for j, val in enumerate(row):
            t33.cell(i+1, j).paragraphs[0].text = val
    style_table(t33, col_widths=[1.5, 1.5, 0.9, 2.8])
    p_cap33 = doc.add_paragraph()
    p_cap33.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_cap33 = p_cap33.add_run("Table 3.3 — Pairwise feature covariance and Pearson correlation")
    r_cap33.font.size = Pt(8.5)
    r_cap33.font.italic = True
    p_cap33.paragraph_format.space_after = Pt(12)

    add_body(
        "The Independence vs. Correlation Distinction: In formal independence testing, P(Demand ∩ Temp) = 0.0938, "
        "while P(Demand) * P(Temp) = 0.0625. The discrepancy of 0.0313 strongly rejects independence. Yet, Pearson correlation is "
        "virtually zero (r = -0.0484). Why? Because Pearson correlation only measures linear relationships. The winter heating "
        "slope is negative, while the summer cooling slope is positive; when integrated across the year, they cancel out linearly "
        "even though the underlying dependency is profound."
    )

    # -------------------------------------------------------------
    # 4. MACHINE LEARNING MODELS
    # -------------------------------------------------------------
    add_heading_1("4. Machine Learning Models")
    
    add_heading_2("4.1 Supervised Learning — Multivariate Polynomial Regression")
    add_body(
        "To avoid data leakage, an 80/20 chronological split was established:\n"
        "• Training Set: First 7,008 hours (January 1 to October 20, 2023)\n"
        "• Test Set: Final 1,752 hours (October 21 to December 31, 2023)\n\n"
        "Baseline linear regression on raw meteorological features underfits severely, yielding a negative test R² (-0.5717). "
        "To solve this within Unit 1 boundaries, domain feature engineering was incorporated:\n"
        "1. Quadratic Temperature Term (Tt²): Captures the thermodynamic U-curve.\n"
        "2. Harmonic Diurnal Encodings: sin(2πH/24) and cos(2πH/24) ensure 23:00 and 00:00 remain close in Euclidean space.\n"
        "3. Calendar Indicators: Weekend binary flags and monthly indicators."
    )
    
    # Table 4.1
    t41 = doc.add_table(rows=3, cols=7)
    headers41 = ["Model Architecture", "Train MAE", "Train RMSE", "Train R²", "Test MAE", "Test RMSE", "Test R²"]
    for j, h in enumerate(headers41):
        t41.cell(0, j).paragraphs[0].text = h
    data41 = [
        ["Baseline Linear Regression", "2,430.43 MW", "3,110.39 MW", "0.5691", "3,797.97 MW", "4,506.56 MW", "-0.5717"],
        ["Feature-Engineered Polynomial", "1,246.30 MW", "1,466.59 MW", "0.9042", "1,332.72 MW", "1,570.08 MW", "0.8092"]
    ]
    for i, row in enumerate(data41):
        for j, val in enumerate(row):
            t41.cell(i+1, j).paragraphs[0].text = val
    style_table(t41, col_widths=[2.1, 0.8, 0.8, 0.7, 0.8, 0.8, 0.7])
    p_cap41 = doc.add_paragraph()
    p_cap41.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_cap41 = p_cap41.add_run("Table 4.1 — Regression performance comparison between baseline and feature-engineered models")
    r_cap41.font.size = Pt(8.5)
    r_cap41.font.italic = True
    p_cap41.paragraph_format.space_after = Pt(12)

    if os.path.exists("figures/10_actual_vs_predicted_demand.png"):
        doc.add_picture("figures/10_actual_vs_predicted_demand.png", width=Inches(5.8))
        p_c10 = doc.add_paragraph()
        p_c10.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_c10 = p_c10.add_run("Figure 4.1 — Actual vs. predicted electricity demand parity plot and 168-hour tracking profile")
        r_c10.font.size = Pt(8.5)
        r_c10.font.italic = True
        p_c10.paragraph_format.space_after = Pt(12)

    add_heading_2("4.2 Unsupervised Learning — K-Means Clustering")
    add_body(
        "K-Means clustering was executed on standardized features [Demand, Solar, Wind, Temperature]. "
        "The elbow method and silhouette analysis identified K=4 as the optimal operational cluster configuration:"
    )
    
    # Table 4.2
    t42 = doc.add_table(rows=5, cols=8)
    headers42 = ["Cluster", "Mean Demand", "Mean Solar", "Mean Wind", "Mean Temp", "Count (n)", "Share (%)", "Grid Operational Regime"]
    for j, h in enumerate(headers42):
        t42.cell(0, j).paragraphs[0].text = h
    data42 = [
        ["0", "27,063 MW", "692 MW", "687 MW", "8.1 °C", "3,236", "36.94%", "Winter Heating Peak (Low Temp, High Load)"],
        ["1", "19,575 MW", "147 MW", "867 MW", "18.0 °C", "2,365", "27.00%", "Off-Peak Baseload / Night Load"],
        ["2", "26,932 MW", "3,060 MW", "696 MW", "25.8 °C", "2,261", "25.81%", "Summer Cooling Peak (High Temp, High Load)"],
        ["3", "23,197 MW", "604 MW", "7,736 MW", "14.3 °C", "898", "10.25%", "Windy Transition (High Wind, Moderate Load)"]
    ]
    for i, row in enumerate(data42):
        for j, val in enumerate(row):
            t42.cell(i+1, j).paragraphs[0].text = val
    style_table(t42, col_widths=[0.6, 1.0, 0.9, 0.9, 0.8, 0.7, 0.7, 1.6])
    p_cap42 = doc.add_paragraph()
    p_cap42.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_cap42 = p_cap42.add_run("Table 4.2 — K-Means discovered grid operational regimes (K = 4)")
    r_cap42.font.size = Pt(8.5)
    r_cap42.font.italic = True
    p_cap42.paragraph_format.space_after = Pt(12)

    if os.path.exists("figures/09_clustering_visualization.png"):
        doc.add_picture("figures/09_clustering_visualization.png", width=Inches(5.8))
        p_c9 = doc.add_paragraph()
        p_c9.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_c9 = p_c9.add_run("Figure 4.2 — K-Means 4-regime clustering visualization on Temperature-Demand plane")
        r_c9.font.size = Pt(8.5)
        r_c9.font.italic = True
        p_c9.paragraph_format.space_after = Pt(12)

    # -------------------------------------------------------------
    # 5. POLYNOMIAL CURVE FITTING
    # -------------------------------------------------------------
    add_heading_1("5. Polynomial Response Modelling and Curve Fitting")
    add_body(
        "Polynomial curve fitting was conducted to model the physical thermodynamic relationship between ambient temperature "
        "and electricity demand. Degrees 1 through 4 were fitted using Ordinary Least Squares normal equations:"
    )
    
    # Table 5.1
    t51 = doc.add_table(rows=5, cols=6)
    headers51 = ["Degree", "Train RMSE", "Train R²", "Test RMSE", "Test R²", "Model Assessment"]
    for j, h in enumerate(headers51):
        t51.cell(0, j).paragraphs[0].text = h
    data51 = [
        ["1 (Linear)", "4,735.27 MW", "0.0014", "3,593.32 MW", "0.0007", "Severe Underfitting (High Bias)"],
        ["2 (Quadratic)", "3,928.05 MW", "0.3128", "4,016.19 MW", "-0.2483", "Captures U-curve, but temp alone lacks calendar factors"],
        ["3 (Cubic)", "3,856.21 MW", "0.3377", "3,946.30 MW", "-0.2052", "Slight improvement in seasonal inflections"],
        ["4 (Quartic)", "3,821.14 MW", "0.3412", "4,210.85 MW", "-0.3210", "Overfitting; high variance at temperature extremes"]
    ]
    for i, row in enumerate(data51):
        for j, val in enumerate(row):
            t51.cell(i+1, j).paragraphs[0].text = val
    style_table(t51, col_widths=[1.0, 1.2, 0.9, 1.2, 0.9, 1.8])
    p_cap51 = doc.add_paragraph()
    p_cap51.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_cap51 = p_cap51.add_run("Table 5.1 — Polynomial degree evaluation on temperature-demand relationship")
    r_cap51.font.size = Pt(8.5)
    r_cap51.font.italic = True
    p_cap51.paragraph_format.space_after = Pt(12)

    if os.path.exists("figures/06_polynomial_regression_curve.png"):
        doc.add_picture("figures/06_polynomial_regression_curve.png", width=Inches(5.8))
        p_c6 = doc.add_paragraph()
        p_c6.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_c6 = p_c6.add_run("Figure 5.1 — Polynomial regression curve fitting (Degrees 1, 2, and 3) on Temperature-Demand data")
        r_c6.font.size = Pt(8.5)
        r_c6.font.italic = True
        p_c6.paragraph_format.space_after = Pt(12)

    # -------------------------------------------------------------
    # 6. SYSTEM ARCHITECTURE
    # -------------------------------------------------------------
    add_heading_1("6. System Architecture")
    add_body(
        "EnergyGrid-AI is organized as a transparent, reproducible, six-stage pipeline:\n\n"
        "DATA INGESTION → PREPROCESSING → STATISTICAL ENGINE → ML MODELS → UQ ENGINE → DASHBOARD\n\n"
        "The project architecture separates raw data, modular mathematical libraries, automated verification tests, "
        "and presentation dashboards into dedicated modules:\n"
        "• data/data_loader.py: Adaptive column matching and CSV loading.\n"
        "• src/preprocessing.py: Imputation, temporal feature engineering, and chronological train/test split.\n"
        "• src/continuous_analysis.py: Unbiased statistics, Gaussian PDF, and KDE density computation.\n"
        "• src/probability_analysis.py: Kolmogorov axiom tests, PMF/CDF, Bayes' rule, and covariance.\n"
        "• src/models.py: Polynomial curve fitting, multivariate linear regression, and K-Means clustering.\n"
        "• src/uncertainty.py: Residual quantile analysis and prediction interval construction.\n"
        "• src/visualizer.py: Matplotlib and Seaborn visualization generator (12 publication figures).\n"
        "• dashboard.py & web_dashboard.py: Operator interfaces for terminal and web browsers."
    )

    # -------------------------------------------------------------
    # 7. IMPLEMENTATION
    # -------------------------------------------------------------
    add_heading_1("7. Implementation")
    add_heading_2("7.1 Technology Stack")
    add_body(
        "The implementation utilizes Python 3.12, NumPy, Pandas, Scikit-learn, SciPy, Matplotlib, Seaborn, and Flask. "
        "Crucially, no deep learning frameworks (TensorFlow, PyTorch) or advanced ensemble libraries (XGBoost, CatBoost) "
        "were used. All algorithms are grounded entirely within linear algebra, classical probability, and statistical theory."
    )
    add_heading_2("7.2 Key Implementation Decisions")
    add_bullet("Chronological Splitting", "Enforcing an 80/20 chronological split (7,008 train / 1,752 test hours) ensures zero future-to-past temporal data leakage.")
    add_bullet("Harmonic Diurnal Encodings", "Trigonometric encoding sin(2πH/24) and cos(2πH/24) models the continuous 24-hour diurnal cycle without artificial midnight discontinuities.")
    add_bullet("Dual Prediction Bounds", "Constructing both non-parametric empirical quantile intervals and parametric Gaussian bounds ensures robust evaluation under skewed residual distributions.")

    # -------------------------------------------------------------
    # 8. EXPERIMENTS AND RESULTS
    # -------------------------------------------------------------
    add_heading_1("8. Experiments and Results")
    add_heading_2("8.1 Uncertainty Quantification Results")
    add_body(
        "Residual errors were evaluated across the 1,752 held-out test hours:\n"
        "• Mean Residual Bias (με): -244.08 MW (negligible relative to 25,000 MW mean load)\n"
        "• Residual Variance (σε²): 2,406,963.70 MW²\n"
        "• Residual Standard Deviation (σε): 1,551.44 MW\n"
        "• Error Quantiles: e0.05 = -2,760.31 MW, e0.50 = -224.66 MW, e0.95 = +2,075.07 MW\n\n"
        "Prediction Interval Performance:\n"
        "1. 90% Empirical Quantile Interval: Average Width = 4,835.38 MW | Empirical Coverage = 89.95% (nominal target 90.00%)\n"
        "2. 95% Gaussian Parametric Interval: Average Width = 6,081.64 MW | Empirical Coverage = 97.32%"
    )

    if os.path.exists("figures/12_prediction_interval_uncertainty.png"):
        doc.add_picture("figures/12_prediction_interval_uncertainty.png", width=Inches(5.8))
        p_c12 = doc.add_paragraph()
        p_c12.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_c12 = p_c12.add_run("Figure 8.1 — Out-of-sample electricity demand forecast with 90% prediction intervals")
        r_c12.font.size = Pt(8.5)
        r_c12.font.italic = True
        p_c12.paragraph_format.space_after = Pt(12)

    # -------------------------------------------------------------
    # 9. DASHBOARD
    # -------------------------------------------------------------
    add_heading_1("9. Dashboard")
    add_body(
        "EnergyGrid-AI provides two distinct user interfaces for operators and academic evaluators:\n"
        "1. Terminal Dashboard (dashboard.py): An ANSI-colored interactive CLI menu enabling rapid inspection of statistics, axioms, model performance, and figure paths.\n"
        "2. Web Dashboard (web_dashboard.py): A comprehensive 8-page web application featuring live interactive sliders for Bayesian risk calculation, interactive time series zoom, parity scatter plots, K-Means cluster profiles, and a technical Viva Voce Q&A module."
    )

    # -------------------------------------------------------------
    # 10. TESTING AND REPRODUCIBILITY
    # -------------------------------------------------------------
    add_heading_1("10. Testing and Reproducibility")
    add_body(
        "Scientific integrity was verified through systematic automated unit tests covering data loading, column adaptation, "
        "duplicate removal, missing value imputation, Kolmogorov axiom consistency, discrete PMF/CDF bounds, Bayesian posterior updates, "
        "feature-engineered regression accuracy, K-Means cluster non-emptiness, and prediction interval coverage probabilities. "
        "All tests execute in under 5 seconds, ensuring complete reproducibility across any standard Python 3.12 environment."
    )

    # -------------------------------------------------------------
    # 11. APPLICATIONS
    # -------------------------------------------------------------
    add_heading_1("11. Applications")
    add_bullet("Transmission System Operator (TSO) Scheduling", "Scheduling day-ahead spinning reserves using calibrated 90% prediction intervals to prevent blackouts.")
    add_bullet("Battery Energy Storage (BESS) Arbitrage", "Optimizing charge/discharge cycles by forecasting net load spikes and renewable generation dips.")
    add_bullet("Renewable Curtailment Reduction", "Predicting periods where solar and wind generation exceed transmission capacity.")
    add_bullet("Dynamic Time-of-Use Tariffs", "Adjusting hourly consumer electricity pricing to incentivize load shifting during high-demand hours.")

    # -------------------------------------------------------------
    # 12. LIMITATIONS
    # -------------------------------------------------------------
    add_heading_1("12. Limitations")
    add_bullet("Linear Model Constraints", "Predictions are restricted to linear combinations of basis functions. Complex interactions beyond quadratic temperature terms are omitted.")
    add_bullet("Static Weather Inputs", "The system assumes deterministic meteorological inputs. In field deployment, numerical weather prediction (NWP) uncertainty compounds forecast error.")
    add_bullet("Single-Year Horizon", "Trained on 8,760 hours of a single calendar year (2023). Multi-year climate shifts and macro-economic demand growth require retraining.")
    add_bullet("Educational Safety Disclaimer", "This is an academic research project developed for B.Tech Machine Learning (Unit 1) and is not certified for direct automated dispatch of physical utility assets.")

    # -------------------------------------------------------------
    # 13. FUTURE SCOPE
    # -------------------------------------------------------------
    add_heading_1("13. Future Scope")
    add_bullet("Autoregressive Lags (Unit 2)", "Incorporating previous-hour demand observations (Dt-1, Dt-24) to capture short-term persistence.")
    add_bullet("Regularization Techniques", "Applying Ridge (L2) and Lasso (L1) regression to prevent overfitting when expanding to higher polynomial degrees.")
    add_bullet("Bayesian Linear Regression", "Inferring analytical posterior distributions over model weights p(w|D) rather than empirical residual quantiles.")
    add_bullet("Dynamic Quantile Regression", "Training pinball loss quantile models to produce heteroscedastic prediction intervals that adapt to extreme weather.")

    # -------------------------------------------------------------
    # 14. CONCLUSION
    # -------------------------------------------------------------
    add_heading_1("14. Conclusion")
    add_body(
        "EnergyGrid-AI demonstrates that an accurate, interpretable, and production-grade energy grid forecasting system can be "
        "constructed strictly within the boundaries of Unit 1 Machine Learning and Probability Theory. By grounding every component "
        "in mathematical rigor—from Kolmogorov addition verification to Bayesian risk updating (1.79× belief update), domain-engineered "
        "polynomial regression (R² = 0.8092, MAE = 1,332.72 MW), unsupervised K-Means regime discovery, and calibrated prediction "
        "intervals (89.95% empirical coverage)—the project bridges classical probability with the real-world operational challenges "
        "of the clean energy transition."
    )

    # -------------------------------------------------------------
    # 15. REFERENCES
    # -------------------------------------------------------------
    add_heading_1("15. References")
    refs = [
        "Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer New York.",
        "Mitchell, T. M. (1997). Machine Learning. McGraw-Hill Education.",
        "Kolmogorov, A. N. (1956). Foundations of the Theory of Probability. Chelsea Publishing Company.",
        "Wood, A. J., Wollenberg, B. F., & Sheblé, G. B. (2013). Power Generation, Operation, and Control. John Wiley & Sons.",
        "Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning: Data Mining, Inference, and Prediction. Springer.",
        "Open Power System Data (2020). Data Package Time series. https://doi.org/10.25832/time_series/"
    ]
    for r in refs:
        p_ref = doc.add_paragraph(style='List Bullet')
        p_ref.paragraph_format.space_after = Pt(4)
        r_run = p_ref.add_run(r)
        r_run.font.name = 'Calibri'
        r_run.font.size = Pt(10)
        r_run.font.color.rgb = RGBColor(51, 65, 85)

    doc.add_page_break()

    # -------------------------------------------------------------
    # APPENDIX: FINAL PROJECT OUTPUTS
    # -------------------------------------------------------------
    p_app_head = doc.add_paragraph()
    r_app_head = p_app_head.add_run("Appendix: Final Project Outputs & Experimental Results")
    r_app_head.font.name = 'Calibri'
    r_app_head.font.size = Pt(20)
    r_app_head.font.bold = True
    r_app_head.font.color.rgb = RGBColor(15, 23, 42)
    p_app_head.paragraph_format.space_after = Pt(12)

    add_heading_2("Executive Summary Metrics")
    t_sum = doc.add_table(rows=11, cols=3)
    headers_sum = ["Performance Metric", "Observed Value", "Operational Significance"]
    for j, h in enumerate(headers_sum):
        t_sum.cell(0, j).paragraphs[0].text = h
    data_sum = [
        ["Total Recorded Hours", "8,760 hours", "Full 365-day chronological year (2023)"],
        ["Supervised Test R²", "0.8092", "Unseen test split accuracy (last 1,752 hours)"],
        ["Supervised Test MAE", "1,332.72 MW", "Mean Absolute Error on test data"],
        ["Supervised Test RMSE", "1,570.08 MW", "Root Mean Squared Error on test data"],
        ["Baseline Test R²", "-0.5717", "Raw linear regression without feature engineering"],
        ["90% Interval Coverage", "89.95%", "Matches nominal 90.00% confidence target"],
        ["Bayesian Risk Update", "1.79×", "Posterior belief multiplier under high temperature"],
        ["Discovered Regimes", "4 Clusters", "K-Means unsupervised operational grid regimes"],
        ["Axiom Discrepancy", "0.000000", "Kolmogorov addition theorem exact verification"],
        ["Operator Dashboards", "8 Web / 7 Terminal", "Full-stack decision support interfaces"]
    ]
    for i, row in enumerate(data_sum):
        for j, val in enumerate(row):
            t_sum.cell(i+1, j).paragraphs[0].text = val
    style_table(t_sum, col_widths=[2.1, 1.8, 2.8])
    
    add_heading_2("Final Research Conclusion")
    add_body(
        "The implemented statistical, probabilistic, supervised, unsupervised, and uncertainty quantification components "
        "consistently identify genuine physical electrical load structures, while maintaining documented operational limits "
        "and safety disclaimers."
    )

    output_path = "ENERGY_GRID_RESEARCH_PROJECT_REPORT.docx"
    doc.save(output_path)
    print(f"Successfully generated: {output_path}")

if __name__ == "__main__":
    create_report()
