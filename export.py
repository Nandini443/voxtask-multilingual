import io
import pandas as pd
from typing import List, Dict, Any

def export_tasks_to_csv(tasks: List[Dict[str, Any]]) -> bytes:
    """Exports list of tasks into CSV bytes."""
    df = pd.DataFrame(tasks)
    # Ensure standard order of columns
    desired_cols = ["action", "owner", "deadline", "priority", "context_snippet"]
    existing_cols = [c for c in desired_cols if c in df.columns]
    if existing_cols:
        df = df[existing_cols]
    
    csv_string = df.to_csv(index=False)
    return csv_string.encode("utf-8")

def export_report_to_pdf(
    meeting_title: str,
    segments: List[Dict[str, Any]],
    tasks: List[Dict[str, Any]]
) -> bytes:
    """Generates a PDF document for the VoxTask meeting summary."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )
        
        styles = getSampleStyleSheet()
        
        # Custom Styles
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#1E1E2F"),
            spaceAfter=4
        )
        
        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#6C757D"),
            spaceAfter=15
        )
        
        section_style = ParagraphStyle(
            "SectionHeader",
            parent=styles["Heading2"],
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#0D6EFD"),
            spaceBefore=15,
            spaceAfter=8
        )
        
        cell_style = ParagraphStyle(
            "TableCell",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#212529")
        )
        
        header_cell_style = ParagraphStyle(
            "HeaderCell",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            fontName="Helvetica-Bold",
            textColor=colors.white
        )

        elements = []
        
        # Title Banner
        elements.append(Paragraph(f"🎙️ VoxTask: {meeting_title}", title_style))
        elements.append(Paragraph("Voice-First Multilingual Meeting & Task Summary Report", subtitle_style))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#DEE2E6"), spaceAfter=15))
        
        # Section 1: Action Items Table
        elements.append(Paragraph("📋 Extracted Action Items", section_style))
        
        table_data = [
            [
                Paragraph("Action Item", header_cell_style),
                Paragraph("Owner", header_cell_style),
                Paragraph("Deadline", header_cell_style),
                Paragraph("Priority", header_cell_style)
            ]
        ]
        
        for task in tasks:
            table_data.append([
                Paragraph(str(task.get("action", "")), cell_style),
                Paragraph(str(task.get("owner", "")), cell_style),
                Paragraph(str(task.get("deadline", "")), cell_style),
                Paragraph(str(task.get("priority", "")), cell_style)
            ])
            
        task_table = Table(table_data, colWidths=[240, 90, 110, 80])
        task_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E1E2F")),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CED4DA")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9FA")])
        ]))
        elements.append(task_table)
        elements.append(Spacer(1, 15))
        
        # Section 2: Meeting Transcript
        elements.append(Paragraph("🗣️ Full Transcript & Speaker Tags", section_style))
        
        transcript_data = [
            [
                Paragraph("Timestamp", header_cell_style),
                Paragraph("Speaker & Language", header_cell_style),
                Paragraph("Transcribed Content", header_cell_style)
            ]
        ]
        
        for seg in segments:
            time_str = f"{seg.get('start', 0.0)}s - {seg.get('end', 0.0)}s"
            speaker_info = f"{seg.get('speaker', 'Speaker')}<br/><font color='#6C757D'>[{seg.get('language', 'En')}]</font>"
            transcript_data.append([
                Paragraph(time_str, cell_style),
                Paragraph(speaker_info, cell_style),
                Paragraph(str(seg.get('text', '')), cell_style)
            ])
            
        trans_table = Table(transcript_data, colWidths=[80, 130, 310])
        trans_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#495057")),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E9ECEF")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9FA")])
        ]))
        elements.append(trans_table)
        
        # Build Document
        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()
        
    except Exception as e:
        print(f"[EXPORT] ReportLab PDF generation failed: {e}")
        # Fallback raw text PDF generation
        fallback_text = f"VoxTask Summary Report\nMeeting: {meeting_title}\n\nTASKS:\n"
        for t in tasks:
            fallback_text += f"- [{t.get('priority')}] {t.get('action')} | Owner: {t.get('owner')} | Deadline: {t.get('deadline')}\n"
        return fallback_text.encode("utf-8")
