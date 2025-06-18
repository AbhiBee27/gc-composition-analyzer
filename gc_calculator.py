import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import re
import io

# --- Page Config ---
st.set_page_config(page_title="AbhiBee’s GC Analyzer", layout="wide")
st.title("🐝 AbhiBee’s GC Composition Analyzer")
st.markdown("Welcome to your mobile and desktop friendly DNA analyzer!")
st.divider()

# --- Sidebar Inputs ---
st.sidebar.header("📥 DNA Input")
uploaded_file = st.sidebar.file_uploader("Upload a FASTA file", type=["fasta", "fa", "txt"])
manual_input = st.sidebar.text_area("Or paste your DNA sequence here:", height=200)
run_analysis = st.sidebar.button("🔍 Analyze DNA")

# --- Session State ---
if "dna" not in st.session_state:
    st.session_state.dna = ""

# --- Read Input ---
if run_analysis:
    dna = ""
    if uploaded_file:
        content = uploaded_file.read().decode("utf-8")
        lines = content.splitlines()
        dna = "".join(line.strip() for line in lines if not line.startswith(">"))
    elif manual_input:
        dna = manual_input.strip()

    dna = dna.upper().replace(" ", "").replace("\n", "")
    valid_bases = {'A', 'T', 'G', 'C', 'N'}
    if not set(dna).issubset(valid_bases):
        st.error("❌ Invalid DNA sequence! Only A, T, G, C, or N are allowed.")
        st.stop()
    else:
        st.session_state.dna = dna

# --- Functions ---
def translate_sequence(seq):
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
        'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
    }
    return "".join([codon_table.get(seq[i:i+3], 'X') for i in range(0, len(seq)-2, 3)])

def find_orfs(seq):
    start_codon = 'ATG'
    stop_codons = {'TAA', 'TAG', 'TGA'}
    orfs = []
    for frame in range(3):
        i = frame
        while i < len(seq) - 2:
            codon = seq[i:i+3]
            if codon == start_codon:
                for j in range(i+3, len(seq)-2, 3):
                    if seq[j:j+3] in stop_codons:
                        orfs.append((i+1, j+3))
                        break
            i += 3
    return orfs

# --- Analysis ---
if st.session_state.dna:
    dna = st.session_state.dna
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
    st.divider()

    st.subheader("📊 Base Composition Pie Chart")
    fig, ax = plt.subplots()
    ax.pie([count_a, count_t, count_g, count_c],
           labels=["A", "T", "G", "C"],
           autopct="%1.1f%%",
           colors=["#FFD700", "#FF6347", "#32CD32", "#1E90FF"])
    ax.axis("equal")
    st.pyplot(fig)
    st.divider()

    window_size = st.slider("Window Size for GC Analysis", min_value=10, max_value=100, value=30, step=5)

    st.subheader("📈 GC% Sliding Window")
    gc_values, positions = [], []
    for i in range(length - window_size + 1):
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
    st.divider()

    st.subheader("📉 GC Skew Plot")
    skew_values = []
    for i in range(length - window_size + 1):
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
    st.divider()

    st.subheader("🌡️ Melting Temperature (Tm)")
    if length <= 14:
        tm = (count_a + count_t) * 2 + (count_g + count_c) * 4
        st.write(f"**Wallace Rule (≤14 bases)** → Tm = {tm}°C")
    else:
        tm = 64.9 + 41 * (count_g + count_c - 16.4) / length
        st.write(f"**Long Sequence Rule (>14 bases)** → Tm ≈ {round(tm, 2)}°C")
    st.divider()

    st.subheader("🔁 Reverse Complement")
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    reverse_comp = ''.join(complement.get(base, 'N') for base in reversed(dna))
    st.code(reverse_comp, language="text")
    st.divider()

    st.subheader("📊 Base Percentage Summary")
    total_counted = count_a + count_t + count_g + count_c + count_n
    base_stats = {
        'A': round((count_a / total_counted) * 100, 2),
        'T': round((count_t / total_counted) * 100, 2),
        'G': round((count_g / total_counted) * 100, 2),
        'C': round((count_c / total_counted) * 100, 2),
        'N': round((count_n / total_counted) * 100, 2)
    }
    st.write(f"A: {base_stats['A']}% | T: {base_stats['T']}% | G: {base_stats['G']}% | C: {base_stats['C']}% | N: {base_stats['N']}%")
    st.divider()

    if count_n > 0:
        st.warning("⚠️ Sequence contains unknown bases (N). These may affect GC% and Tm calculations.")

    st.subheader("🔍 Motif Search")
    motif = st.text_input("Enter motif to search (e.g. ATG, A[AT]G):")
    if motif:
        try:
            matches = [m.start() + 1 for m in re.finditer(motif.upper(), dna)]
            if matches:
                st.success(f"✅ Found {len(matches)} matches at positions: {matches}")
            else:
                st.info("No matches found.")
        except re.error:
            st.error("❌ Invalid motif pattern.")
    st.divider()

    st.subheader("🧬 ORF Finder")
    orfs = find_orfs(dna)
    if orfs:
        st.success(f"✅ Found {len(orfs)} ORFs:")
        df_orf = pd.DataFrame(orfs, columns=["Start", "End"])
        df_orf["Length"] = df_orf["End"] - df_orf["Start"]
        df_orf["ORF Sequence"] = df_orf.apply(lambda row: dna[row.Start-1:row.End], axis=1)
        st.dataframe(df_orf, use_container_width=True)
        st.download_button("⬇️ Download ORFs CSV", df_orf.to_csv(index=False), "orfs.csv", "text/csv")
    else:
        st.info("No ORFs found.")
    st.divider()

    st.subheader("🔠 DNA → Protein Translation")
    protein = translate_sequence(dna)
    st.code(protein, language="text")
    st.divider()

    st.subheader("💾 Export Results")
    export_data = {
        "Length": length,
        "GC_Content": gc_content,
        "A_Count": count_a,
        "T_Count": count_t,
        "G_Count": count_g,
        "C_Count": count_c,
        "N_Count": count_n,
        "A%": base_stats['A'],
        "T%": base_stats['T'],
        "G%": base_stats['G'],
        "C%": base_stats['C'],
        "N%": base_stats['N'],
        "Tm": round(tm, 2),
        "Reverse_Complement": reverse_comp,
        "Protein": protein
    }
    df_export = pd.DataFrame([export_data])
    csv = df_export.to_csv(index=False)
    txt = df_export.to_string(index=False)
    st.download_button("⬇️ Download CSV", csv, "gc_analysis.csv", "text/csv")
    st.download_button("⬇️ Download TXT", txt, "gc_analysis.txt", "text/plain")
    st.divider()
else:
    st.info("📋 Paste your sequence or upload a file, then click '🔍 Analyze DNA'")
