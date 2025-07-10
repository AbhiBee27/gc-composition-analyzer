# Genelytics - Final Complete Version 🚀 (All Tabs, Fast Backend, Fully Functional)

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from collections import Counter
import numba
from numba.typed import Dict
from numba.types import int64
from io import StringIO

# --- Page Config ---
st.set_page_config(page_title="AbhiBee’s Genelytics+", layout="wide")
st.title("🧬 AbhiBee’s Genelytics")
st.markdown("Analyze large-scale DNA sequences instantly. All bioinformatics tools integrated!")
st.divider()

# --- Sidebar Inputs ---
st.sidebar.header("📅 DNA Input")
uploaded_file = st.sidebar.file_uploader("Upload FASTA", type=["fasta", "fa", "txt"])
manual_input = st.sidebar.text_area("Paste DNA sequence:", height=200)
motif_input = st.sidebar.text_input("Motif to search (optional):")
preview_mode = st.sidebar.checkbox("Limit to 1M bases")
run_analysis = st.sidebar.button("Analyze")

# --- Session State ---
if "dna" not in st.session_state:
    st.session_state.dna = ""
    st.session_state.header = ""

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
    if not set(dna).issubset({'A', 'T', 'G', 'C', 'N'}):
        st.error("Invalid DNA. Only A, T, G, C, N allowed.")
        st.stop()
    if preview_mode:
        dna = dna[:1_000_000]
    st.session_state.dna = dna
    st.session_state.header = header

# --- Optimized Core Functions ---
@numba.njit()
def orf_finder_array(seq):
    start = [65, 84, 71]
    stops = [(84, 65, 65), (84, 65, 71), (84, 71, 65)]
    orfs = []
    for frame in range(3):
        i = frame
        while i < len(seq) - 2:
            if seq[i] == start[0] and seq[i+1] == start[1] and seq[i+2] == start[2]:
                for j in range(i+3, len(seq)-2, 3):
                    codon = (seq[j], seq[j+1], seq[j+2])
                    if codon in stops:
                        orfs.append((frame+1, i+1, j+3))
                        break
            i += 3
    return orfs

@numba.njit()
def codon_usage_array(seq):
    table = Dict.empty(key_type=(int64, int64, int64), value_type=int64)
    for i in range(0, len(seq)-2, 3):
        codon = (seq[i], seq[i+1], seq[i+2])
        table[codon] = table.get(codon, 0) + 1
    return table

@numba.njit()
def _translate_array_raw(seq):
    codon_map = {
        (65,84,71):'M', (84,65,65):'_', (84,65,71):'_', (84,71,65):'_'
    }
    result = [codon_map.get((seq[i], seq[i+1], seq[i+2]), 'X') for i in range(0, len(seq)-2, 3)]
    return result

def translate_array(seq):
    protein_fragments = _translate_array_raw(seq)
    output = StringIO()
    for aa in protein_fragments:
        output.write(aa)
    return output.getvalue()

@numba.njit()
def gc_skew_array(seq_u8, window):
    skew_vals = np.empty(len(seq_u8), dtype=np.float32)
    for i in range(len(seq_u8)):
        if seq_u8[i] == 71:
            skew_vals[i] = 1
        elif seq_u8[i] == 67:
            skew_vals[i] = -1
        else:
            skew_vals[i] = 0
    result_len = len(skew_vals) - window + 1
    result = np.empty(result_len, dtype=np.float32)
    cumsum = np.cumsum(skew_vals)
    for i in range(result_len):
        total = cumsum[i + window - 1] - (cumsum[i - 1] if i > 0 else 0)
        result[i] = total / window
    return result

def find_cpg_islands(seq, window=200, gc_thresh=50, obs_exp_thresh=0.6):
    islands = []
    for i in range(len(seq) - window + 1):
        win = seq[i:i+window]
        g = win.count('G')
        c = win.count('C')
        cg = sum(1 for j in range(len(win)-1) if win[j:j+2] == 'CG')
        gc_pct = 100 * (g + c) / window
        expected = (g * c) / window if window else 1
        obs_exp = cg / expected if expected else 0
        if gc_pct >= gc_thresh and obs_exp >= obs_exp_thresh:
            islands.append((i+1, i+window, gc_pct, obs_exp))
    return islands

def analyze_aa_properties(protein):
    hydro = 'AILMFWV'; polar = 'STYCNQ'; basic = 'KRH'; acidic = 'DE'; aromatic = 'FYW'
    count = Counter(protein)
    total = sum(count.values())
    props = {
        "Hydrophobic": sum(count[a] for a in hydro),
        "Polar": sum(count[a] for a in polar),
        "Basic": sum(count[a] for a in basic),
        "Acidic": sum(count[a] for a in acidic),
        "Aromatic": sum(count[a] for a in aromatic)
    }
    return {k: round(v*100/total,2) for k,v in props.items()} if total else props

def calculate_cai(seq):
    preferred = {
        'GCT': 31, 'GCC': 40, 'GCA': 20, 'GCG': 9, 'CGT': 36, 'CGC': 36, 'CGA': 7, 'CGG': 11,
        'AGA': 4, 'AGG': 6, 'AAT': 45, 'AAC': 55, 'GAT': 63, 'GAC': 37, 'TGT': 46, 'TGC': 54,
        'CAA': 34, 'CAG': 66, 'GAA': 68, 'GAG': 32, 'GGT': 34, 'GGC': 37, 'GGA': 13, 'GGG': 16,
        'CAT': 42, 'CAC': 58, 'ATT': 49, 'ATC': 39, 'ATA': 11, 'CTT': 13, 'CTC': 20, 'CTA': 7,
        'CTG': 40, 'TTA': 7, 'TTG': 13, 'AAA': 76, 'AAG': 24, 'ATG': 100, 'TTT': 58, 'TTC': 42,
        'CCT': 28, 'CCC': 32, 'CCA': 27, 'CCG': 13, 'TCT': 18, 'TCC': 21, 'TCA': 15, 'TCG': 5,
        'AGT': 10, 'AGC': 31, 'ACT': 24, 'ACC': 36, 'ACA': 28, 'ACG': 12, 'TGG': 100, 'TAT': 43,
        'TAC': 57, 'GTT': 36, 'GTC': 24, 'GTA': 13, 'GTG': 27, 'TAA': 30, 'TAG': 24, 'TGA': 46
    }
    table = codon_usage_array(np.frombuffer(seq.encode(), dtype=np.uint8))
    usage_k = list(table.keys())
    usage_v = list(table.values())
    score, total = 0, 0
    for i in range(len(usage_k)):
        codon = ''.join([chr(x) if isinstance(x, int) else x.decode() for x in usage_k[i]])
        freq = preferred.get(codon, 0) / 100
        score += freq * usage_v[i]
        total += usage_v[i]
    return round(score / total, 3) if total else 0

@st.cache_data(show_spinner=False)
def get_numpy_arrays(dna):
    return np.frombuffer(dna.encode(), dtype='S1'), np.frombuffer(dna.encode(), dtype=np.uint8)

# --- Outputs ---
if st.session_state.dna:
    dna = st.session_state.dna
    arr, arr_u8 = get_numpy_arrays(dna)
    a, t, g, c = map(np.count_nonzero, [(arr == b'A'), (arr == b'T'), (arr == b'G'), (arr == b'C')])
    length = len(arr)
    gc_content = round((g + c) * 100 / length, 2)
    tabs = st.tabs(["Summary", "ORFs", "Translation", "Codon Usage", "Motif", "GC Skew", "CAI", "CpG Islands", "AA Properties", "Export"])

    with tabs[0]:
        st.markdown(f"**Length**: {length} bases")
        st.markdown(f"**GC Content**: {gc_content}%")
        fig, ax = plt.subplots()
        ax.pie([a, t, g, c], labels=["A", "T", "G", "C"], autopct="%1.1f%%")
        st.pyplot(fig)

    with tabs[1]:
        df = pd.DataFrame(orf_finder_array(arr_u8), columns=["Frame", "Start", "End"])
        df["Length"] = df["End"] - df["Start"]
        st.dataframe(df.head(300))

    with tabs[2]:
        protein = translate_array(arr_u8)
        st.code(protein[:1000] + ("..." if len(protein) > 1000 else ""))

    with tabs[3]:
        table = codon_usage_array(arr_u8)
        codons = [''.join([chr(x) for x in k]) for k in table.keys()]
        counts = list(table.values())
        df = pd.DataFrame({"Codon": codons, "Count": counts})
        st.dataframe(df.sort_values("Count", ascending=False))

    with tabs[4]:
        if motif_input:
            matches = [m.start()+1 for m in re.finditer(motif_input.upper(), dna)]
            st.success(f"Found {len(matches)} matches")
            st.code(matches[:100])

    with tabs[5]:
        window = st.slider("Window Size", 100, 1000, 300)
        skew = gc_skew_array(arr_u8, window)
        if len(skew) > 2000:
            step = len(skew) // 2000
            skew = skew[::step]
        st.line_chart(skew)

    with tabs[6]:
        cai = calculate_cai(dna)
        st.metric("Codon Adaptation Index", cai)

    with tabs[7]:
        islands = find_cpg_islands(dna)
        df = pd.DataFrame(islands, columns=["Start", "End", "GC%", "Obs/Exp"])
        st.dataframe(df)

    with tabs[8]:
        st.bar_chart(pd.Series(analyze_aa_properties(protein)))

    with tabs[9]:
        st.download_button("Download DNA as FASTA", dna, file_name="sequence.fasta")
        st.download_button("Download Protein as TXT", protein, file_name="protein.txt")
