# AbhiBee’s Genelytics+ (Optimized Final Version)
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from collections import Counter
import numba
import plotly.graph_objects as go

# --- Page Setup ---
st.set_page_config(page_title="AbhiBee’s Genelytics+", layout="wide")
st.title("🧬 AbhiBee’s Genelytics")
st.markdown("Smart scoring, mutation simulation, GC insights — now super fast.")
st.divider()

# --- Sidebar Inputs ---
st.sidebar.header("📅 DNA Input")
uploaded_file = st.sidebar.file_uploader("Upload FASTA", type=["fasta", "fa", "txt"])
manual_input = st.sidebar.text_area("Paste DNA sequence:", height=200)
motif_input = st.sidebar.text_input("Motif to search (optional):")
run_analysis = st.sidebar.button("Analyze")

# --- Session State ---
if "dna" not in st.session_state:
    st.session_state.dna = ""
    st.session_state.header = ""

if run_analysis:
    dna, header = "", ""
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

    st.session_state.dna = dna
    st.session_state.header = header

# --- Utilities ---
@st.cache_data
def get_numpy_arrays(dna):
    return np.frombuffer(dna.encode(), dtype='S1'), np.frombuffer(dna.encode(), dtype=np.uint8)

@st.cache_data
def get_gc_skew(dna: str, window: int):
    arr_u8 = np.frombuffer(dna.encode(), dtype=np.uint8)
    skew_vals = np.where(arr_u8 == 71, 1, np.where(arr_u8 == 67, -1, 0))
    result = np.convolve(skew_vals, np.ones(window), 'valid') / window
    return result






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

@st.cache_data
def get_translation_and_usage(dna):
    codon_map = {
        'AAA':'K','AAC':'N','AAG':'K','AAT':'N','ACA':'T','ACC':'T','ACG':'T','ACT':'T',
        'AGA':'R','AGC':'S','AGG':'R','AGT':'S','ATA':'I','ATC':'I','ATG':'M','ATT':'I',
        'CAA':'Q','CAC':'H','CAG':'Q','CAT':'H','CCA':'P','CCC':'P','CCG':'P','CCT':'P',
        'CGA':'R','CGC':'R','CGG':'R','CGT':'R','CTA':'L','CTC':'L','CTG':'L','CTT':'L',
        'GAA':'E','GAC':'D','GAG':'E','GAT':'D','GCA':'A','GCC':'A','GCG':'A','GCT':'A',
        'GGA':'G','GGC':'G','GGG':'G','GGT':'G','GTA':'V','GTC':'V','GTG':'V','GTT':'V',
        'TAA':'_','TAC':'Y','TAG':'_','TAT':'Y','TCA':'S','TCC':'S','TCG':'S','TCT':'S',
        'TGA':'_','TGC':'C','TGG':'W','TGT':'C','TTA':'L','TTC':'F','TTG':'L','TTT':'F'
    }
    dna = dna.upper()
    codons = [dna[i:i+3] for i in range(0, len(dna)-2, 3)]
    protein = [codon_map.get(c, 'X') for c in codons]
    return "".join(protein), dict(Counter(codons))

@st.cache_data
def get_cai_score(usage: dict):
    preferred = {'ATG': 100, 'TTT': 92, 'TTC': 80, 'TTA': 40, 'TTG': 50}
    score, total = 0, 0
    for codon, count in usage.items():
        freq = preferred.get(codon, 0) / 100
        score += freq * count
        total += count
    return round(score / total, 3) if total else 0

@st.cache_data
def score_orfs_batch(dna: str, orfs: list):
    codon_map = {
        'AAA':'K','AAC':'N','AAG':'K','AAT':'N','ACA':'T','ACC':'T','ACG':'T','ACT':'T',
        'AGA':'R','AGC':'S','AGG':'R','AGT':'S','ATA':'I','ATC':'I','ATG':'M','ATT':'I',
        'CAA':'Q','CAC':'H','CAG':'Q','CAT':'H','CCA':'P','CCC':'P','CCG':'P','CCT':'P',
        'CGA':'R','CGC':'R','CGG':'R','CGT':'R','CTA':'L','CTC':'L','CTG':'L','CTT':'L',
        'GAA':'E','GAC':'D','GAG':'E','GAT':'D','GCA':'A','GCC':'A','GCG':'A','GCT':'A',
        'GGA':'G','GGC':'G','GGG':'G','GGT':'G','GTA':'V','GTC':'V','GTG':'V','GTT':'V',
        'TAA':'_','TAC':'Y','TAG':'_','TAT':'Y','TCA':'S','TCC':'S','TCG':'S','TCT':'S',
        'TGA':'_','TGC':'C','TGG':'W','TGT':'C','TTA':'L','TTC':'F','TTG':'L','TTT':'F'
    }
    preferred = {'ATG': 100, 'TTT': 92, 'TTC': 80, 'TTA': 40, 'TTG': 50}
    results = []

    for frame, start, end in orfs:
        seq = dna[start - 1:end]
        codons = [seq[i:i+3] for i in range(0, len(seq)-2, 3)]
        protein = [codon_map.get(c, 'X') for c in codons]
        aa_seq = ''.join(protein)
        usage = Counter(codons)
        length = len(aa_seq)

        if length < 30:
            score = 20.0
        else:
            cai_score, total = 0, 0
            for codon, count in usage.items():
                freq = preferred.get(codon, 0) / 100
                cai_score += freq * count
                total += count
            cai = round(cai_score / total, 3) if total else 0

            gc = (seq.count('G') + seq.count('C')) * 100 / len(seq)
            internal_stops = aa_seq.count('_') - 1
            rare_penalty = sum(1 for c in usage if usage[c] < 3)

            score = (length / 10) + (cai * 40) - (internal_stops * 5) - (rare_penalty * 0.5)
            if 40 <= gc <= 65:
                score += 5
            if seq.startswith("ATGG"):
                score += 5
            score = round(min(100, max(0, score)), 1)

        results.append((frame, start, end, end - start, score))

    return results






@numba.njit()
def find_cpg_islands(seq_u8, window=200, gc_thresh=50.0, obs_exp_thresh=0.6):
    islands = []
    for i in range(len(seq_u8) - window + 1):
        g, c, cg = 0, 0, 0
        for j in range(window):
            base = seq_u8[i + j]
            if base == 71: g += 1
            elif base == 67: c += 1
            if j < window - 1 and seq_u8[i + j] == 67 and seq_u8[i + j + 1] == 71:
                cg += 1
        gc_pct = 100 * (g + c) / window
        expected = (g * c) / window
        obs_exp = cg / expected if expected else 0
        if gc_pct >= gc_thresh and obs_exp >= obs_exp_thresh:
            islands.append((i+1, i+window, gc_pct, obs_exp))
    return islands

def analyze_aa_properties(protein):
    hydro = 'AILMFWV'; polar = 'STYCNQ'; basic = 'KRH'; acidic = 'DE'; aromatic = 'FYW'
    count = Counter(protein)
    total = sum(count.values())
    return {
        "Hydrophobic": round(sum(count[a] for a in hydro) * 100 / total, 2),
        "Polar": round(sum(count[a] for a in polar) * 100 / total, 2),
        "Basic": round(sum(count[a] for a in basic) * 100 / total, 2),
        "Acidic": round(sum(count[a] for a in acidic) * 100 / total, 2),
        "Aromatic": round(sum(count[a] for a in aromatic) * 100 / total, 2),
    }

def simulate_mutation_impact(orf_seq: str):
    codon_map = {
        'AAA':'K','AAC':'N','AAG':'K','AAT':'N','ACA':'T','ACC':'T','ACG':'T','ACT':'T',
        'AGA':'R','AGC':'S','AGG':'R','AGT':'S','ATA':'I','ATC':'I','ATG':'M','ATT':'I',
        'CAA':'Q','CAC':'H','CAG':'Q','CAT':'H','CCA':'P','CCC':'P','CCG':'P','CCT':'P',
        'CGA':'R','CGC':'R','CGG':'R','CGT':'R','CTA':'L','CTC':'L','CTG':'L','CTT':'L',
        'GAA':'E','GAC':'D','GAG':'E','GAT':'D','GCA':'A','GCC':'A','GCG':'A','GCT':'A',
        'GGA':'G','GGC':'G','GGG':'G','GGT':'G','GTA':'V','GTC':'V','GTG':'V','GTT':'V',
        'TAA':'_','TAC':'Y','TAG':'_','TAT':'Y','TCA':'S','TCC':'S','TCG':'S','TCT':'S',
        'TGA':'_','TGC':'C','TGG':'W','TGT':'C','TTA':'L','TTC':'F','TTG':'L','TTT':'F'
    }
    bases = ['A', 'T', 'C', 'G']
    impact = []

    for i in range(0, len(orf_seq)-2, 3):
        codon = orf_seq[i:i+3]
        if len(codon) != 3:
            continue
        original_aa = codon_map.get(codon, 'X')
        for pos in range(3):
            for b in bases:
                if b != codon[pos]:
                    mutated = list(codon)
                    mutated[pos] = b
                    mut_codon = ''.join(mutated)
                    mut_aa = codon_map.get(mut_codon, 'X')
                    if mut_aa == original_aa:
                        impact_type = 'Synonymous'
                    elif mut_aa == '_':
                        impact_type = 'Nonsense'
                    elif mut_aa == 'X':
                        continue
                    else:
                        impact_type = 'Missense'
                    impact.append({
                        "Codon Index": i//3 + 1,
                        "Original Codon": codon,
                        "Mutated Codon": mut_codon,
                        "Original AA": original_aa,
                        "Mutated AA": mut_aa,
                        "Type": impact_type
                    })
    return impact







# --- MAIN APP UI ---
if st.session_state.dna:
    dna = st.session_state.dna
    arr, arr_u8 = get_numpy_arrays(dna)
    a, t, g, c = map(np.count_nonzero, [(arr == b'A'), (arr == b'T'), (arr == b'G'), (arr == b'C')])
    length = len(arr)
    gc_content = round((g + c) * 100 / length, 2)
    protein, usage = get_translation_and_usage(dna)
    cai = get_cai_score(usage)

    raw_orfs = orf_finder_array(arr_u8)

    # ✅ Limit to 500 ORFs for speed
    with st.spinner("Scoring top ORFs..."):
        scored_orfs = score_orfs_batch(dna, raw_orfs[:500])

    tabs = st.tabs([
        "Summary", "ORFs", "Translation", "Codon Usage", "Motif",
        "GC Skew", "CAI", "CpG Islands", "AA Properties", "Mutation Tolerance", "Export"
    ])

    with tabs[0]:
        st.markdown(f"**Length**: {length:,} bases")
        st.markdown(f"**GC Content**: {gc_content}%")
        fig, ax = plt.subplots()
        ax.pie([a, t, g, c], labels=["A", "T", "G", "C"], autopct="%1.1f%%")
        st.pyplot(fig)

    with tabs[1]:
        df = pd.DataFrame(scored_orfs, columns=["Frame", "Start", "End", "Length", "Confidence Score"])
        st.dataframe(df.sort_values("Confidence Score", ascending=False).reset_index(drop=True))

    with tabs[2]:
        st.code(protein[:1000] + ("..." if len(protein) > 1000 else ""))
        st.caption(f"Full protein length: {len(protein):,} amino acids")

    with tabs[3]:
        df = pd.DataFrame({"Codon": list(usage.keys()), "Count": list(usage.values())})
        st.dataframe(df.sort_values("Count", ascending=False))

    with tabs[4]:
        if motif_input:
            matches = [m.start()+1 for m in re.finditer(motif_input.upper(), dna)]
            st.success(f"Found {len(matches)} matches")
            st.code(matches[:100])

    with tabs[5]:
        max_window = min(len(dna)//5, 1000)
        window = st.slider("Window Size", 50, max_window, 300, key="gc_window_slider")
        with st.spinner("Calculating GC Skew..."):
            skew = get_gc_skew(dna, window)
        positions = np.arange(len(skew)) + window // 2
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=positions, y=skew, mode='lines', name='GC Skew'))
        fig.update_layout(title="GC Skew Plot", xaxis_title="Position", yaxis_title="GC Skew", height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[6]:
        st.metric("Codon Adaptation Index", cai)

    with tabs[7]:
        islands = find_cpg_islands(arr_u8)
        df = pd.DataFrame(islands, columns=["Start", "End", "GC%", "Obs/Exp"])
        st.dataframe(df)

    with tabs[8]:
        st.bar_chart(pd.Series(analyze_aa_properties(protein)))

    with tabs[9]:
        st.subheader("Mutation Tolerance for Top ORF")
        if scored_orfs:
            top_orf = max(scored_orfs, key=lambda x: x[-1])
            orf_seq = dna[top_orf[1]-1:top_orf[2]]
            impact = simulate_mutation_impact(orf_seq)
            df_mut = pd.DataFrame(impact)
            st.dataframe(df_mut)
        else:
            st.warning("No ORFs detected to evaluate.")

    with tabs[10]:
        st.download_button("Download DNA as FASTA", dna, file_name="sequence.fasta")
        st.download_button("Download Protein as TXT", protein, file_name="protein.txt")
