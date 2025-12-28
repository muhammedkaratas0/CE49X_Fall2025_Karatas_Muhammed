import os
import sys
from fpdf import FPDF
from datetime import datetime

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
output_dir = os.path.join(project_root, "output", "v2")



class PDF(FPDF):
    def header(self):
        # Register Font (Must be done once)
        if not hasattr(self, 'font_registered'):
            # Use System Arial Font
            font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'
            bold_font_path = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
            
            # Check if bold exists, otherwise reuse regular
            if not os.path.exists(bold_font_path):
                bold_font_path = font_path

            self.add_font('ArialCustom', '', font_path, uni=True)
            self.add_font('ArialCustom', 'B', bold_font_path, uni=True)
            self.font_registered = True
            
        self.set_font('ArialCustom', '', 8)
        self.cell(0, 10, 'Civil Engineering & AI Integration - Final Analysis Report', 0, 1, 'R')
        self.line(10, 20, 200, 20)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('ArialCustom', '', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

    def chapter_title(self, label):
        self.set_font('ArialCustom', 'B', 16)
        self.set_fill_color(230, 230, 250) # Light lavender
        self.cell(0, 12, label, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, text):
        self.set_font('ArialCustom', '', 11)
        self.multi_cell(0, 6, text)
        self.ln()

    def add_image_centered(self, image_path, width=170):
        if os.path.exists(image_path):
            self.image(image_path, x=(210-width)/2, w=width)
            self.ln(5)
        else:
            self.set_text_color(255, 0, 0)
            self.cell(0, 10, f"Image not found: {os.path.basename(image_path)}", 0, 1)
            self.set_text_color(0, 0, 0)

    def add_bullet_point(self, text):
        self.set_font('ArialCustom', '', 11)
        self.cell(10) # Indent
        self.cell(5, 6, chr(149), 0, 0, 'R') # Bullet char (or just use '-')
        self.multi_cell(0, 6, text)
        self.ln(2)

def create_report():
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # --- TITLE PAGE ---
    pdf.ln(50)
    # Ensure font is set before use (redundant but safe)
    if not hasattr(pdf, 'font_registered'):
        font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'
        bold_font_path = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
        if not os.path.exists(bold_font_path): bold_font_path = font_path
        
        pdf.add_font('ArialCustom', '', font_path, uni=True)
        pdf.add_font('ArialCustom', 'B', bold_font_path, uni=True)
        pdf.font_registered = True
    
    pdf.set_font('ArialCustom', 'B', 24)
    pdf.cell(0, 20, "Civil Engineering & AI", 0, 1, 'C')
    pdf.set_font('ArialCustom', 'B', 18)
    pdf.cell(0, 15, "Strategic Analysis of Integration Trends", 0, 1, 'C')
    
    pdf.ln(30)
    pdf.set_font('ArialCustom', '', 14)
    pdf.cell(0, 10, "Prepared By:", 0, 1, 'C')
    pdf.set_font('ArialCustom', 'B', 14)
    pdf.cell(0, 10, "Muhammed Ali Karataş", 0, 1, 'C')
    pdf.cell(0, 10, "Bora Işık", 0, 1, 'C')
    
    pdf.ln(30)
    pdf.set_font('ArialCustom', '', 12)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%B %d, %Y')}", 0, 1, 'C')
    
    pdf.add_page()

    # --- 1. MOTIVATION ("Why We Did It") ---
    pdf.chapter_title("1. Motivation & Objective")
    pdf.chapter_body(
        "The Civil Engineering (CE) sector is undergoing a rapid digital transformation driven by "
        "Artificial Intelligence (AI). However, the adoption landscape is fragmented. "
        "Our objective was to demystify this integration by analyzing real-world data."
    )
    pdf.ln(2)
    pdf.add_bullet_point("Identify which CE domains are successfully adopting AI.")
    pdf.add_bullet_point("Determine the specific AI technologies driving this change.")
    pdf.add_bullet_point("Provide a data-driven roadmap for future research and investment.")
    pdf.ln(5)

    # --- 2. METHODOLOGY ("How We Did It") ---
    pdf.chapter_title("2. Methodology")
    pdf.chapter_body(
        "We built a custom automated pipeline to aggregate, process, and visualize global trends "
        "in CE and AI."
    )
    pdf.ln(2)
    pdf.set_font('ArialCustom', 'B', 11)
    pdf.cell(0, 8, "Step 1: Data Collection", 0, 1)
    pdf.chapter_body(
        "We aggregated 744 articles from diverse sources including Google News, RSS feeds, and academic repositories "
        "to ensure a comprehensive dataset."
    )
    
    pdf.set_font('ArialCustom', 'B', 11)
    pdf.cell(0, 8, "Step 2: AI-Powered Classification", 0, 1)
    pdf.chapter_body(
        "We utilized Google Gemini Pro (LLM) to intelligently analyze each article. "
        "The model extracted key metadata: Primary CE Area, specific AI Technologies, and Sentiment Scores."
    )

    pdf.set_font('ArialCustom', 'B', 11)
    pdf.cell(0, 8, "Step 3: Strategic Visualization", 0, 1)
    pdf.chapter_body(
        "Using Python libraries (Pandas, Plotly, NetworkX), we generated advanced visualizations "
        "to highlight correlations and clusters."
    )
    pdf.ln(5)

    pdf.add_page()

    # --- 3. KEY FINDINGS ("Bulgular") ---
    pdf.chapter_title("3. Key Findings (Bulgular)")
    
    # Finding 1
    pdf.set_font('ArialCustom', 'B', 12)
    pdf.cell(0, 10, "Finding 1: Construction Management Leads Adoption", 0, 1)
    pdf.chapter_body(
        "Our analysis reveals that Construction Management is the most active area for AI integration. "
        "It accounts for the highest volume of diverse AI applications, focusing on schedule optimization "
        "and safety monitoring."
    )
    pdf.add_image_centered(os.path.join(output_dir, "1_bar_chart_ce_areas_v2.png"))
    
    # Finding 2
    pdf.add_page()
    pdf.set_font('ArialCustom', 'B', 12)
    pdf.cell(0, 10, "Finding 2: Computer Vision is the Dominant Tech", 0, 1)
    pdf.chapter_body(
        "The heatmap below demonstrates a strong clustering of Computer Vision applications within "
        "Construction Management and Structural Engineering (specifically for defect detection). "
        "This indicates a shift towards visual data processing."
    )
    pdf.add_image_centered(os.path.join(output_dir, "2_heatmap_ce_ai_v2.png"))

    # Finding 3
    pdf.add_page()
    pdf.set_font('ArialCustom', 'B', 12)
    pdf.cell(0, 10, "Finding 3: Interdisciplinary Hubs", 0, 1)
    pdf.chapter_body(
        "The Network Graph exposes the 'connective tissue' of the industry. We observe that "
        "Machine Learning acts as a central hub connecting almost all CE disciplines, "
        "while specialized techs like Robotics are more isolated to specific tasks."
    )
    pdf.add_image_centered(os.path.join(output_dir, "3_network_graph_v2.png"), width=180)

    # Finding 4
    pdf.add_page()
    pdf.set_font('ArialCustom', 'B', 12)
    pdf.cell(0, 10, "Finding 4: Thematic Focus Areas", 0, 1)
    pdf.chapter_body(
        "Word cloud analysis highlights practical implementation keywords. For Construction Management, "
        "terms like 'Safety', 'Monitoring', and 'Optimization' are prevalent."
    )
    pdf.add_image_centered(os.path.join(output_dir, "4_wordcloud_construction_management_v2.png"))

    # --- CONCLUSION ---
    pdf.add_page()
    pdf.chapter_title("4. Conclusion")
    pdf.chapter_body(
        "This study confirms that AI is no longer just a theoretical concept in Civil Engineering but "
        "an active driver of efficiency and safety. The dominance of Construction Management and "
        "Computer Vision suggests that the immediate value of AI is currently found in "
        "monitoring and optimizing physical processes. Future opportunities lie in expanding these "
        "technologies to Geotechnical and Environmental challenges."
    )
    
    # Save
    output_path = os.path.join(project_root, "Final_Report.pdf")
    pdf.output(output_path)
    print(f"Report generated successfully: {output_path}")

if __name__ == "__main__":
    try:
        create_report()
    except Exception as e:
        print(f"Error generating PDF: {e}")
