import streamlit as st
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(page_title="AbhiBee’s GC Analyzer", layout="wide")
st.title("🐝 AbhiBee’s GC Composition Analyzer")
st.markdown("Welcome to your mobile and desktop friendly DNA analyzer!")

# --- Sidebar Inputs ---
st.sidebar.header("📥 DNA Input")
uploaded_file = st.sidebar.file_uploader("Upload a FASTA file", type=["fasta", "fa", "txt"])
manual_input = st.sidebar.text_area("Or paste your DNA sequence here:", height=200)
run_analysis = st.sidebar.button("🔍 Analyze DNA")

# --- Session State to Store DNA ---
if "dna" not in st.session_state:
    st.session_state.dna = ""

# --- Read Input Sequence ---
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
        st.session_state.dna = dna  # Store for future reruns

# --- Use Stored DNA for Analysis ---
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

    # --- Pie Chart ---
    st.subheader("📊 Base Composition Pie Chart")
    fig, ax = plt.subplots()
    ax.pie([count_a, count_t, count_g, count_c],
           labels=["A", "T", "G", "C"],
           autopct="%1.1f%%",
           colors=["#FFD700", "#FF6347", "#32CD32", "#1E90FF"])
    ax.axis("equal")
    st.pyplot(fig)
    st.divider()

    # --- GC% and GC Skew Slider ---
    window_size = st.slider("Window Size for GC Analysis", min_value=10, max_value=100, value=30, step=5)

    # --- GC% Sliding Window ---
    st.subheader("📈 GC% Sliding Window")
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
    st.divider()

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
    st.divider()

    # --- Melting Temperature (Tm) ---
    st.subheader("🌡️ Melting Temperature (Tm)")
    if length <= 14:
        tm = (count_a + count_t) * 2 + (count_g + count_c) * 4
        st.write(f"**Wallace Rule (≤14 bases)** → Tm = {tm}°C")
    else:
        tm = 64.9 + 41 * (count_g + count_c - 16.4) / length
        st.write(f"**Long Sequence Rule (>14 bases)** → Tm ≈ {round(tm, 2)}°C")
    st.divider()

    # --- Reverse Complement ---
    st.subheader("🔁 Reverse Complement")
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    reverse_comp = ''.join(complement.get(base, 'N') for base in reversed(dna))
    st.code(reverse_comp, language="text")
    st.divider()

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
else:
    st.info("📋 Paste your sequence or upload a file, then click '🔍 Analyze DNA'")
