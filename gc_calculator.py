import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF

# --- Page Config ---
st.set_page_config(page_title="AbhiBee’s GC Analyzer", layout="wide")
st.title("🐝 AbhiBee’s GC Composition Analyzer")
st.markdown("Welcome to your mobile and desktop friendly DNA analyzer!")

# --- Sidebar Inputs ---
st.sidebar.header("📥 DNA Input")
uploaded_file = st.sidebar.file_uploader("Upload a FASTA file", type=["fasta", "fa", "txt"])
manual_input = st.sidebar.text_area("Or paste your DNA sequence here:", height=200)
motif = st.sidebar.text_input("🔎 Enter motif to search (optional):")
run_analysis = st.sidebar.button("🔍 Analyze DNA")

# --- Read Input Sequence ---
dna = ""
if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    lines = content.splitlines()
    dna = "".join(line.strip() for line in lines if not line.startswith(">"))
elif manual_input:
    dna = manual_input.strip()

# --- When Analyze Button is Clicked ---
if run_analysis and dna:
    dna = dna.upper().replace(" ", "").replace("\n", "")
    valid_bases = {'A', 'T', 'G', 'C', 'N'}

    if not set(dna).issubset(valid_bases):
        st.error("❌ Invalid DNA sequence! Only A, T, G, C, or N are allowed.")
    else:
        # --- Base Counts ---
        length = len(dna)
        count_a = dna.count('A')
        count_t = dna.count('T')
        count_g = dna.count('G')
        count_c = dna.count('C')
        count_n = dna.count('N')
        gc_content = round((count_g + count_c) * 100 / length, 2)

        st.header("🔎 Sequence Summary")
        st.markdown(f"""
        - **Length:** {length} bases  
        - **GC Content:** {gc_content}%  
        - **A:** {count_a} | **T:** {count_t} | **G:** {count_g} | **C:** {count_c} | **N:** {count_n}
        """)

        # --- Pie Chart ---
        st.subheader("📊 Base Composition Pie Chart")
        fig, ax = plt.subplots()
        ax.pie([count_a, count_t, count_g, count_c],
               labels=["A", "T", "G", "C"],
               autopct="%1.1f%%",
               colors=["#FFD700", "#FF6347", "#32CD32", "#1E90FF"])
        ax.axis("equal")
        st.pyplot(fig)

        # --- Sliding Window GC% ---
        st.subheader("📈 GC% Sliding Window")
        window_size = st.slider("Window Size", min_value=10, max_value=100, value=30, step=5)
        gc_values = []
        positions = []

        for i in range(len(dna) - window_size + 1):
            window = dna[i:i + window_size]
            g = window.count('G')
            c = window.count('C')
            gc = (g + c) * 100 / window_size
            gc_values.append(gc)
            positions.append(i + 1)

        fig2, ax2 = plt.subplots()
        ax2.plot(positions, gc_values, color='green')
        ax2.set_xlabel("Position")
        ax2.set_ylabel("GC%")
        ax2.set_title("GC% Sliding Window")
        st.pyplot(fig2)

        # --- GC Skew ---
        st.subheader("📉 GC Skew Plot")
        skew_values = []

        for i in range(len(dna) - window_size + 1):
            window = dna[i:i + window_size]
            g = window.count('G')
            c = window.count('C')
            skew = (g - c) / (g + c) if (g + c) > 0 else 0
            skew_values.append(skew)

        fig3, ax3 = plt.subplots()
        ax3.plot(positions, skew_values, color='purple')
        ax3.set_xlabel("Position")
        ax3.set_ylabel("GC Skew")
        ax3.set_title("GC Skew Plot")
        st.pyplot(fig3)

        # --- Melting Temperature (Tm) ---
        st.subheader("🌡️ Melting Temperature (Tm)")
        if length <= 14:
            tm = (count_a + count_t) * 2 + (count_g + count_c) * 4
            st.write(f"**Wallace Rule (≤14 bases)** → Tm = {tm}°C")
        else:
            tm = 64.9 + 41 * (count_g + count_c - 16.4) / length
            st.write(f"**Long Sequence Rule (>14 bases)** → Tm ≈ {round(tm, 2)}°C")

        # --- Reverse Complement ---
        st.subheader("🔁 Reverse Complement")
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
        reverse_comp = ''.join(complement.get(base, 'N') for base in reversed(dna))
        st.code(reverse_comp, language="text")

        # --- Base Percentages ---
        st.subheader("📊 Base Percentage Summary")
        base_stats = {
            'A': round((count_a / length) * 100, 2),
            'T': round((count_t / length) * 100, 2),
            'G': round((count_g / length) * 100, 2),
            'C': round((count_c / length) * 100, 2),
            'N': round((count_n / length) * 100, 2)
        }
        st.write(f"A: {base_stats['A']}% | T: {base_stats['T']}% | G: {base_stats['G']}% | C: {base_stats['C']}% | N: {base_stats['N']}%")

        if count_n > 0:
            st.warning("⚠️ Sequence contains unknown bases (N). These may affect GC% and Tm calculations.")

        # --- Protein Translation ---
        st.subheader("🧬 DNA to Protein Translation")
        codon_table = {
            'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
            'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
            'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
            'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
            'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
            'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
            'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
            'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
            'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
            'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
            'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
            'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
            'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
            'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
            'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
            'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W'
        }
        protein = ''
        for i in range(0, len(dna) - 2, 3):
            codon = dna[i:i+3]
            protein += codon_table.get(codon, 'X')
        st.code(protein)

        # --- Motif Finder ---
        if motif:
            st.subheader("🔎 Motif Search")
            motif = motif.upper()
            matches = [i+1 for i in range(len(dna) - len(motif) + 1) if dna[i:i+len(motif)] == motif]
            if matches:
                st.success(f"Motif found at positions: {matches}")
            else:
                st.warning("Motif not found in sequence.")
        else:
            matches = None

        # --- Export Options (Bottom) ---
        st.subheader("📁 Export Results")
        txt_content = f"""DNA Length: {length}
GC Content: {gc_content}%
A: {count_a}, T: {count_t}, G: {count_g}, C: {count_c}, N: {count_n}
GC% Sliding Window Size: {window_size}
Melting Temp: {tm}
Reverse Complement: {reverse_comp}
Base Percentages: {base_stats}
Protein: {protein}
Motif: {motif if motif else 'None'}
Matches: {matches if motif else 'N/A'}
"""
        st.download_button("💾 Download as TXT", data=txt_content, file_name="gc_analysis.txt")

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for line in txt_content.splitlines():
            pdf.cell(200, 10, txt=line, ln=1)
        pdf_output = pdf.output(dest='S').encode('latin1')
        st.download_button("📄 Download as PDF", data=pdf_output, file_name="gc_analysis.pdf", mime="application/pdf")

else:
    st.info("📋 Paste your sequence or upload a file, then click '🔍 Analyze DNA'")
