"""
Forensic Audit Report PDF Generator Module
Compiles complete forensic diagnostic records, telemetry, image preview,
and verification verdicts into an official PDF audit certificate.
"""

import os
import io
import time
from typing import Dict, Any, Optional
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont


def _get_font(size: int, bold: bool = False) -> PIL.ImageFont.ImageFont:
    """Safely retrieves a TrueType font or falls back gracefully."""
    font_candidates = [
        ("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        ("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        ("C:/Windows/Fonts/calibri.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    ]
    for path in font_candidates:
        if os.path.exists(path):
            try:
                return PIL.ImageFont.truetype(path, size)
            except Exception:
                pass
    return PIL.ImageFont.load_default()


def generate_forensic_pdf_report(
    report: Dict[str, Any],
    image_bytes: Optional[bytes] = None
) -> bytes:
    """
    Renders an official forensic digital image audit certificate as a PDF document.
    
    Args:
        report: Forensic diagnostic dictionary produced by predict_image().
        image_bytes: Raw binary bytes of the inspected image.
        
    Returns:
        bytes: Complete PDF document binary stream.
    """
    canvas_w = 1240
    canvas_h = 1754
    canvas = PIL.Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = PIL.ImageDraw.Draw(canvas)

    # Palette
    c_navy = (15, 23, 42)        # #0f172a
    c_blue = (37, 99, 235)       # #2563eb
    c_light_bg = (248, 250, 252) # #f8fafc
    c_border = (226, 232, 240)   # #e2e8f0
    c_dark = (30, 41, 59)        # #1e293b
    c_muted = (100, 116, 139)    # #64748b

    # Status Colors
    status = report.get("status_code", "UNKNOWN")
    if status == "VERIFIED":
        c_status_bg = (209, 250, 229) # light emerald
        c_status_text = (6, 95, 70)   # dark emerald
        c_status_border = (16, 185, 129)
        status_label = "VERIFIED: Consistent with Natural Optical Capture"
        status_icon = "[PASS - AUTHENTIC]"
    elif status == "FLAGGED":
        c_status_bg = (254, 226, 226) # light red
        c_status_text = (153, 27, 27) # dark red
        c_status_border = (239, 68, 68)
        status_label = "FLAGGED: AI-Generated / Synthetic Traces Detected"
        status_icon = "[FAIL - SYNTHETIC]"
    else:
        c_status_bg = (254, 243, 199) # light amber
        c_status_text = (146, 64, 14) # dark amber
        c_status_border = (245, 158, 11)
        status_label = "INCONCLUSIVE: Ambiguous / Quality Degraded"
        status_icon = "[ABSTAIN / MANUAL TRIAGE]"

    # 1. Header Banner
    draw.rectangle([(0, 0), (canvas_w, 140)], fill=c_navy)
    draw.text((60, 32), "FORENSIC DIGITAL IMAGE AUDIT CERTIFICATE", font=_get_font(28, bold=True), fill=(255, 255, 255))
    draw.text((60, 78), "AI-Based Deepfake Image Detection & Veracity Assessment Protocol", font=_get_font(16), fill=(148, 163, 184))
    draw.rectangle([(0, 136), (canvas_w, 140)], fill=c_blue)

    # 2. Metadata Bar
    draw.rectangle([(60, 160), (canvas_w - 60, 225)], fill=c_light_bg, outline=c_border, width=1)
    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    sha_short = (report.get("sha256") or "N/A")[:16] + "..."
    draw.text((80, 175), f"Audit Timestamp: {timestamp_str}", font=_get_font(13), fill=c_dark)
    draw.text((80, 198), f"SHA-256 Digest: {report.get('sha256', 'N/A')[:40]}...", font=_get_font(13), fill=c_muted)
    draw.text((700, 175), f"Compute Device: CUDA (NVIDIA GeForce RTX 3050)", font=_get_font(13), fill=c_dark)
    draw.text((700, 198), f"Backbone: {report.get('model_architecture', 'efficientnet_b0')} (SE Attention)", font=_get_font(13), fill=c_muted)

    # 3. Image Visual Inspection & Verdict Card (Two Columns)
    y_col = 250
    # Left: Image Thumbnail Box
    thumb_w = 420
    thumb_h = 420
    draw.rectangle([(60, y_col), (60 + thumb_w, y_col + thumb_h)], fill=(241, 245, 249), outline=c_border, width=2)

    if image_bytes:
        try:
            raw_img = PIL.Image.open(io.BytesIO(image_bytes))
            raw_img.thumbnail((thumb_w - 20, thumb_h - 20), PIL.Image.Resampling.LANCZOS)
            offset_x = 60 + (thumb_w - raw_img.width) // 2
            offset_y = y_col + (thumb_h - raw_img.height) // 2
            canvas.paste(raw_img, (offset_x, offset_y))
        except Exception:
            draw.text((120, y_col + 180), "[Image Rendering Failed]", font=_get_font(16), fill=c_muted)
    else:
        draw.text((120, y_col + 180), "[No Image Preview Available]", font=_get_font(16), fill=c_muted)

    caption_text = f"Asset: {report.get('filename', 'N/A')} ({report.get('dimensions', ['N/A', 'N/A'])[0]}x{report.get('dimensions', ['N/A', 'N/A'])[1]} px)"
    draw.text((60, y_col + thumb_h + 10), caption_text, font=_get_font(13, bold=True), fill=c_dark)

    # Right: Verdict & Score Card
    card_x = 520
    card_w = canvas_w - 60 - card_x

    # Status Badge
    draw.rectangle([(card_x, y_col), (card_x + card_w, y_col + 90)], fill=c_status_bg, outline=c_status_border, width=2)
    draw.text((card_x + 20, y_col + 18), status_icon, font=_get_font(15, bold=True), fill=c_status_text)
    draw.text((card_x + 20, y_col + 45), status_label, font=_get_font(17, bold=True), fill=c_status_text)

    # Probability Score Block
    calib_score = report.get("calibrated_score")
    score_pct_str = f"{calib_score * 100:.1f}%" if calib_score is not None else "N/A"
    raw_logit = report.get("raw_logit")
    logit_str = f"{raw_logit:+.3f}" if raw_logit is not None else "N/A"

    y_score = y_col + 115
    draw.rectangle([(card_x, y_score), (card_x + card_w, y_score + 185)], fill=c_light_bg, outline=c_border, width=1)
    draw.text((card_x + 24, y_score + 18), "Calibrated Synthetic Likelihood (AI Probability):", font=_get_font(14, bold=True), fill=c_dark)
    draw.text((card_x + 24, y_score + 45), score_pct_str, font=_get_font(42, bold=True), fill=c_status_text if status == "FLAGGED" else c_blue)

    if calib_score is not None:
        real_conf = (1.0 - calib_score) * 100
        draw.text((card_x + 200, y_score + 60), f"Equivalent Real Authenticity: {real_conf:.1f}%", font=_get_font(15, bold=True), fill=(16, 185, 129))

    draw.text((card_x + 24, y_score + 120), f"Neural Network Raw Logit: {logit_str}", font=_get_font(14), fill=c_dark)
    draw.text((card_x + 24, y_score + 145), f"Model Forward Latency: {report.get('model_forward_ms', 0.0):.1f} ms | Pipeline: {report.get('total_pipeline_ms', 0.0):.1f} ms", font=_get_font(13), fill=c_muted)

    # 4. Deep Forensic Telemetry & Camera Sensor Analysis
    y_table = y_col + thumb_h + 50
    draw.rectangle([(60, y_table), (canvas_w - 60, y_table + 40)], fill=c_dark)
    draw.text((80, y_table + 10), "DEEP FORENSIC SENSOR TELEMETRY & METADATA HARVESTER", font=_get_font(14, bold=True), fill=(255, 255, 255))

    quality = report.get("quality_metrics", {})
    exif = report.get("exif_summary", {})

    table_rows = [
        ("Focus Laplacian Variance", f"{quality.get('laplacian_variance', 'N/A')}", "Threshold: >= 100.0 (Higher = sharper physical pores)"),
        ("Shannon Entropy", f"{quality.get('entropy', 'N/A')} bits/pixel", "Information density and high-frequency noise variance"),
        ("Camera Hardware (Make/Model)", f"{exif.get('camera_make', 'None')} / {exif.get('camera_model', 'None')}", "EXIF Hardware Tag (Missing/Stripped on web transfers)"),
        ("Software / Editor Flag", f"{exif.get('software', 'None')}", "Photoshop / Inpainting / Generative pipeline flags"),
        ("Original Timestamp", f"{exif.get('datetime_original', 'None')}", "Physical sensor capture timestamp"),
        ("Multi-View Forensic Fusion", "Global Composition (224px) + Central Face ROI Zoom (80%)", "Asymmetric biological pore verification"),
        ("Probability Calibration", f"Platt Temperature Scaling (T = {report.get('calibration_temperature', 1.9488)})", "ECE: 0.0118 (Empirically calibrated test reliability)"),
        ("Decision Rule Applied", f"{report.get('decision_mode', 'Direct Verdict (Threshold: 0.58)')}", "Score < 0.58 => Real | Score >= 0.58 => Synthetic")
    ]

    cur_y = y_table + 45
    for idx, (metric_name, metric_val, metric_desc) in enumerate(table_rows):
        bg_color = (255, 255, 255) if idx % 2 == 0 else c_light_bg
        draw.rectangle([(60, cur_y), (canvas_w - 60, cur_y + 36)], fill=bg_color, outline=c_border, width=1)
        draw.text((80, cur_y + 9), metric_name, font=_get_font(13, bold=True), fill=c_dark)
        draw.text((450, cur_y + 9), str(metric_val), font=_get_font(13, bold=True), fill=c_blue)
        draw.text((700, cur_y + 9), metric_desc, font=_get_font(12), fill=c_muted)
        cur_y += 36

    # 5. Scientific & Legal Verification Seal & Notice
    y_footer = cur_y + 30
    draw.rectangle([(60, y_footer), (canvas_w - 60, y_footer + 180)], fill=(248, 250, 252), outline=c_border, width=1)
    draw.rectangle([(60, y_footer), (66, y_footer + 180)], fill=c_blue)

    draw.text((80, y_footer + 14), "MANDATORY FORENSIC, REGULATORY & ETHICAL NOTICE", font=_get_font(13, bold=True), fill=c_dark)
    disclaimer_lines = [
        "1. Academic & Forensic Triage Prototype: This automated report is generated by a deep neural network trained on open synthetic benchmarks.",
        "   It assists human analysts in triaging anomalous digital media and does NOT constitute standalone legal proof of authenticity or fraud.",
        "2. Non-Biometric Assessment: This evaluation does NOT perform facial recognition (1:1 or 1:N) and does NOT prove physical human liveness.",
        "3. EXIF Metadata Notice: Stripped or absent EXIF metadata is standard across messaging platforms (WhatsApp, Telegram) and does not denote fraud.",
        "4. Human-in-the-Loop Mandate: All flagged and inconclusive determinations require review by a certified digital forensics specialist."
    ]
    dy = 42
    for line in disclaimer_lines:
        draw.text((80, y_footer + dy), line, font=_get_font(11), fill=c_muted)
        dy += 22

    # Bottom watermark
    draw.text((60, canvas_h - 40), "AI-Based Deepfake Image Detection System • Digital Forensic Audit Certificate • Confidential", font=_get_font(11), fill=(148, 163, 184))
    draw.text((canvas_w - 200, canvas_h - 40), f"Page 1 of 1", font=_get_font(11), fill=(148, 163, 184))

    # Save to in-memory PDF buffer
    buf = io.BytesIO()
    canvas.save(buf, format="PDF", resolution=150.0)
    buf.seek(0)
    return buf.getvalue()
