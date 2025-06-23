# Optimized version of AbhiBee's Genelytics for large-scale DNA input

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import re
import io
import requests
from collections import Counter
import numpy as np
import numba
from concurrent.futures import ThreadPoolExecutor

# --- Page Config ---
st.set_page_config(page_title="AbhiBee’s Genelytics", layout="wide")
st.title("🧬 AbhiBee’s Genelytics")
st.markdown("Welcome to your mobile and desktop friendly DNA analyzer!")
st.divider()

# --- Sidebar Inputs ---
st.sidebar.header("📅 DNA Input")
uploaded_file = st.sidebar.file_uploader("Upload a FASTA file", type=["fasta", "fa", "txt"])
manual_input = st.sidebar.text_area("Or paste your DNA sequence here:", height=200)
motif_input = st.sidebar.text_input("🔎 Enter motif to search (optional):")
preview_mode = st.sidebar.checkbox("🧩 Preview mode (limit to 1M bases)")
run_analysis = st.sidebar.button("🔍 Analyze DNA")

# --- Session State ---
if "dna" not in st.session_state:
    st.session_state.dna = ""
    st.session_state.header = ""

# --- Read Input ---
if run_analysis:
    dna = ""
    header = ""
    if uploaded_file:
        content = uploaded_file.read().decode("utf-8")
        lines = content.splitlines()
        header = lines[0][1:] if lines and lines[0].startswith(">") else ""
        dna = "".join(line.strip() for line in lines if not line.startswith(">"))
    elif manual_input:
        dna = manual_input.strip()

    dna = dna.upper().replace(" ", "").replace("\n", "")
    valid_bases = {'A', 'T', 'G', 'C', 'N'}
    if not set(dna).issubset(valid_bases):
        st.error("❌ Invalid DNA sequence! Only A, T, G, C, or N are allowed.")
        st.stop()
    if preview_mode:
        dna = dna[:1_000_000]
    st.session_state.dna = dna
    st.session_state.header = header

# --- Optimized Functions ---
def translate_sequence(seq):
    table = {
        'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K', 'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V', 'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E', 'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
        'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_', 'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W'
    }
    output = []
    for i in range(0, len(seq)-2, 3):
        codon = seq[i:i+3]
        output.append(table.get(codon, 'X'))
    return "".join(output)

def get_codon_usage(dna):
    return Counter(dna[i:i+3] for i in range(0, len(dna)-2, 3))

def find_orfs(seq, max_orfs=1000):
    start_codon = 'ATG'
    stop_codons = {'TAA', 'TAG', 'TGA'}
    orfs = []
    for frame in range(3):
        i = frame
        while i < len(seq) - 2:
            if seq[i:i+3] == start_codon:
                for j in range(i+3, len(seq)-2, 3):
                    if seq[j:j+3] in stop_codons:
                        orfs.append((frame+1, i+1, j+3))
                        break
            if len(orfs) >= max_orfs:
                break
            i += 3
        if len(orfs) >= max_orfs:
            break
    return orfs

# --- Output ---
if st.session_state.dna:
    dna = st.session_state.dna
    header = st.session_state.header

    st.subheader("📌 FASTA Header")
    st.info(header if header else "No header found.")
    st.divider()

    st.subheader("🔎 Sequence Summary")
    arr = np.frombuffer(dna.encode(), dtype='S1')
    count_a = np.count_nonzero(arr == b'A')
    count_t = np.count_nonzero(arr == b'T')
    count_g = np.count_nonzero(arr == b'G')
    count_c = np.count_nonzero(arr == b'C')
    count_n = np.count_nonzero(arr == b'N')
    length = len(arr)
    gc_content = round((count_g + count_c) * 100 / length, 2)
    st.markdown(f"""
    - **Length:** {length} bases  
    - **GC Content:** {gc_content}%  
    - **A:** {count_a} | **T:** {count_t} | **G:** {count_g} | **C:** {count_c} | **N:** {count_n}
    """)
    st.divider()

    st.subheader("📊 Base Composition Pie Chart")
    fig, ax = plt.subplots()
    ax.pie([count_a, count_t, count_g, count_c, count_n],
           labels=["A", "T", "G", "C", "N"],
           autopct="%1.1f%%",
           colors=["#FFD700", "#FF6347", "#32CD32", "#1E90FF", "#999999"])
    ax.axis("equal")
    st.pyplot(fig)
    st.divider()

    st.subheader("📈 GC% Sliding Window (with Downsampling)")
    window_size = st.slider("Window Size", min_value=10, max_value=100, value=30, step=5)
    is_gc = np.isin(arr, [b'G', b'C']).astype(int)
    gc_sliding = np.convolve(is_gc, np.ones(window_size), 'valid') * 100 / window_size
    positions = np.arange(1, len(gc_sliding) + 1)
    if len(gc_sliding) > 1000:
        step = len(gc_sliding) // 1000
        gc_sliding = gc_sliding[::step]
        positions = positions[::step]
    fig, ax = plt.subplots()
    ax.plot(positions, gc_sliding, color='green')
    ax.set_title("GC% Sliding Window")
    st.pyplot(fig)
    st.divider()

    st.subheader("🔁 Reverse Complement")
    comp_map = {b'A': 'T', b'T': 'A', b'G': 'C', b'C': 'G', b'N': 'N'}
    rev = arr[::-1]
    rev_comp = ''.join([comp_map.get(base, 'N') for base in rev.tolist()])
    st.code(rev_comp[:10000] + ("..." if len(rev_comp) > 10000 else ""))
    st.divider()

    st.subheader("🌡️ Melting Temperature")
    tm = (count_a + count_t) * 2 + (count_g + count_c) * 4 if length <= 14 else 64.9 + 41 * (count_g + count_c - 16.4) / length
    st.write(f"Estimated Tm: {round(tm, 2)}°C")
    st.divider()

    st.subheader("🔍 Motif Search")
    if motif_input:
        try:
            motif_pattern = re.compile(motif_input.upper())
            matches = [m.start()+1 for m in motif_pattern.finditer(dna)]
            if matches:
                st.success(f"Found {len(matches)} match(es) at positions: {matches[:10]}{'...' if len(matches) > 10 else ''}")
                with st.expander("📄 View All Matches"):
                    st.code(", ".join(map(str, matches)))
            else:
                st.warning("No matches found.")
        except re.error:
            st.error("Invalid motif pattern.")
    st.divider()

    st.subheader("🧬 DNA to Protein Translation")
    protein = translate_sequence(dna)
    st.code(protein[:10000] + ("..." if len(protein) > 10000 else ""))
    st.divider()

    st.subheader("🌈 Amino Acid Composition")
    aa_counts = Counter(protein)
    aa_df = pd.DataFrame(aa_counts.items(), columns=["Amino Acid", "Count"]).sort_values("Count", ascending=False)
    st.dataframe(aa_df, use_container_width=True)
    st.divider()

    st.subheader("🧪 Codon Usage Frequency")
    codon_counts = get_codon_usage(dna)
    codon_df = pd.DataFrame(codon_counts.items(), columns=["Codon", "Count"]).sort_values("Count", ascending=False)
    st.dataframe(codon_df, use_container_width=True)
    st.divider()

    st.subheader("📊 ORF Finder & Visualization")
    orfs = find_orfs(dna)
    if orfs:
        df_orf = pd.DataFrame(orfs, columns=["Frame", "Start", "End"])
        df_orf["Length"] = df_orf["End"] - df_orf["Start"]
        st.dataframe(df_orf, use_container_width=True)
        fig, ax = plt.subplots(figsize=(10, 2))
        for i, (frame, start, end) in enumerate(orfs[:200]):
            ax.broken_barh([(start, end-start)], (frame*2, 1.5), facecolors='tab:blue')
        ax.set_ylim(0, 8)
        ax.set_xlabel("Position")
        ax.set_title("ORF Visualization (Frames 1-3)")
        st.pyplot(fig)
    else:
        st.info("No ORFs found.")
    st.divider()

    st.subheader("🔬 NCBI Gene Identifier (BLAST RID)")
    def blast_request():
        try:
            response = requests.post("https://blast.ncbi.nlm.nih.gov/Blast.cgi", data={
                'CMD': 'Put',
                'PROGRAM': 'blastp',
                'DATABASE': 'nr',
                'QUERY': protein[:300]
            })
            rid = re.search(r"RID = (\w+)", response.text)
            if rid:
                st.success(f"BLAST request submitted. RID: {rid.group(1)}")
            else:
                st.error("Unable to extract RID from BLAST response.")
        except:
            st.error("BLAST request failed. Check your internet connection or try again later.")

    if st.button("🧬 Run NCBI BLASTP"):
        with ThreadPoolExecutor():
            blast_request()
    st.divider()

    st.subheader("📀 Export Results")
    export_data = {
        "Length": length,
        "GC_Content": gc_content,
        "Tm": round(tm, 2),
        "Reverse_Complement": rev_comp[:100] + "...",
        "Protein": protein[:100] + "..."
    }
    df_export = pd.DataFrame([export_data])
    csv = df_export.to_csv(index=False)
    txt = df_export.to_string(index=False)
    st.download_button("⬇️ Download CSV", csv, "gc_analysis.csv", "text/csv")
    st.download_button("⬇️ Download TXT", txt, "gc_analysis.txt", "text/plain")
    st.divider()
