import streamlit as st
import pandas as pd
import re
from difflib import SequenceMatcher
import io

# Page config
st.set_page_config(page_title="Nepali Corpus Corrector", layout="wide", page_icon="📝")

# Custom CSS for better UI
st.markdown("""
<style>
    .stTextArea textarea {
        font-size: 18px !important;
        font-family: 'Mangal', 'Devanagari', 'Arial', sans-serif !important;
    }
    .good-match {
        background-color: #d4edda;
        padding: 5px;
        border-radius: 5px;
    }
    .bad-match {
        background-color: #f8d7da;
        padding: 5px;
        border-radius: 5px;
    }
    .progress-text {
        font-size: 14px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'df' not in st.session_state:
    st.session_state.df = None
if 'edits_log' not in st.session_state:
    st.session_state.edits_log = []
if 'auto_save' not in st.session_state:
    st.session_state.auto_save = True

def similarity_score(a, b):
    """Calculate string similarity for warning system"""
    return SequenceMatcher(None, a, b).ratio()

def detect_romanized_issues(romanized, devanagari):
    """Detect common LLM hallucination patterns"""
    issues = []
    
    # Common wrong mappings (you can expand this)
    wrong_mappings = {
        'paryawaran': 'पर्यावरण',  # Should map to this, not वातावरण
        'bhawishya': 'भविष्य',
        'sansar': 'संसार',
        'project':'प्रोजेक्ट',
        
    }
    
    for wrong, correct_deva in wrong_mappings.items():
        if wrong in romanized.lower():
            # Check if Devanagari has वातावरण instead of पर्यावरण
            if wrong == 'paryawaran' and 'वातावरण' in devanagari:
                issues.append(f"⚠️ '{wrong}' should become '{correct_deva}', not '{devanagari}'")
    
    # Length ratio check (very rough heuristic)
    len_ratio = len(romanized) / len(devanagari) if devanagari else 0
    if len_ratio < 0.3 or len_ratio > 3:
        issues.append(f"⚠️ Unusual length ratio: {len_ratio:.2f} (romanized:{len(romanized)} vs devanagari:{len(devanagari)})")
    
    return issues

def load_csv(uploaded_file):
    """Load CSV with proper encoding handling"""
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
        # Ensure required columns exist
        if 'romanized' not in df.columns or 'devanagari' not in df.columns:
            st.error("CSV must have 'romanized' and 'devanagari' columns")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

# Sidebar controls
with st.sidebar:
    st.header("📂 Load Corpus")
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        if st.session_state.df is None:
            st.session_state.df = load_csv(uploaded_file)
            if st.session_state.df is not None:
                st.success(f"Loaded {len(st.session_state.df)} sentences")
    
    if st.session_state.df is not None:
        st.header("⚙️ Controls")
        
        # Navigation
        col1, col2 = st.columns(2)
        with col1:
            if st.button("◀ Previous", use_container_width=True):
                if st.session_state.current_index > 0:
                    st.session_state.current_index -= 1
                    st.rerun()
        with col2:
            if st.button("Next ▶", use_container_width=True):
                if st.session_state.current_index < len(st.session_state.df) - 1:
                    st.session_state.current_index += 1
                    st.rerun()
        
        # Jump to specific row
        jump_to = st.number_input("Jump to row", min_value=1, 
                                   max_value=len(st.session_state.df), 
                                   value=st.session_state.current_index + 1)
        if st.button("Go"):
            st.session_state.current_index = jump_to - 1
            st.rerun()
        
        st.divider()
        
        # Filters
        st.header("🔍 Quick Filters")
        
        # Only show problematic rows
        if st.button("Show Only Flagged Rows", use_container_width=True):
            # This requires pre-computed flags - we'll do it on the fly
            st.session_state.filter_flagged = True
            st.rerun()
        
        if st.button("Show All Rows", use_container_width=True):
            st.session_state.filter_flagged = False
            st.rerun()
        
        st.divider()
        
        # Save options
        st.header("💾 Save Progress")
        
        if st.button("Export Corrected CSV", use_container_width=True):
            csv = st.session_state.df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="corrected_corpus.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.success("Ready to download!")

# Main content
st.title("📝 Nepali Corpus Corrector")
st.caption("Interactive tool for correcting romanized → Devanagari parallel corpus")

if st.session_state.df is not None:
    # Get current row
    idx = st.session_state.current_index
    total = len(st.session_state.df)
    current_row = st.session_state.df.iloc[idx]
    
    # Progress bar
    progress = (idx + 1) / total
    st.progress(progress)
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"<span class='progress-text'>Row {idx + 1} of {total} ({progress*100:.1f}% complete)</span>", 
                   unsafe_allow_html=True)
    with col2:
        if st.button("⏭ Skip (keep current)"):
            if idx < total - 1:
                st.session_state.current_index += 1
                st.rerun()
    
    # Flag row for later review
    with col3:
        if st.button("🚩 Flag for later"):
            st.toast("Row flagged!", icon="🚩")
            # You could store flags in session state
    
    st.divider()
    
    # Display similarity warning
    similarity = similarity_score(current_row['romanized'], current_row['devanagari'])
    if similarity < 0.3:
        st.warning(f"⚠️ Low similarity ({similarity:.2%}) - This pair might have major issues")
    
    # Detect known issues
    issues = detect_romanized_issues(current_row['romanized'], current_row['devanagari'])
    for issue in issues:
        st.error(issue)
    
    # Editable fields side by side
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.subheader("🔤 Romanized")
        new_romanized = st.text_area(
            "Edit romanized text",
            value=current_row['romanized'],
            height=150,
            key=f"romanized_{idx}",
            help="Edit the romanized Nepali text here"
        )
        
        # Suggestions for common fixes
        if 'paryawaran' in new_romanized.lower():
            st.info("💡 Did you mean 'paryawaran' → पर्यावरण? Note: LLMs often wrongly map this to वातावरण")
    
    with col2:
        st.subheader("📖 Devanagari")
        new_devanagari = st.text_area(
            "Edit Devanagari text",
            value=current_row['devanagari'],
            height=150,
            key=f"devanagari_{idx}",
            help="Edit the Devanagari text here"
        )
        
        # Quick character insert buttons for Devanagari
        st.caption("Quick insert: ")
        dev_chars = ['ा', 'ि', 'ी', 'े', 'ै', 'ो', 'ौ', 'ं', 'ः', '्', '।', '॥']
        char_cols = st.columns(12)
        for i, char in enumerate(dev_chars):
            if char_cols[i % 12].button(char, key=f"char_{idx}_{i}"):
                new_devanagari += char
                st.rerun()
    
    # Action buttons
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("✅ Save Changes", type="primary", use_container_width=True):
            # Save changes to dataframe
            st.session_state.df.at[idx, 'romanized'] = new_romanized
            st.session_state.df.at[idx, 'devanagari'] = new_devanagari
            
            # Log edit
            st.session_state.edits_log.append({
                'row': idx,
                'old_romanized': current_row['romanized'],
                'new_romanized': new_romanized,
                'old_devanagari': current_row['devanagari'],
                'new_devanagari': new_devanagari
            })
            
            st.success(f"✅ Saved! {len(st.session_state.edits_log)} total edits made")
            
            # Auto-advance to next row
            if idx < total - 1:
                st.session_state.current_index += 1
                st.rerun()
    
    with col2:
        if st.button("↺ Reset current row", use_container_width=True):
            st.rerun()  # Reloads original from dataframe
    
    with col3:
        if st.button("📋 Copy to clipboard", use_container_width=True):
            st.write("Copy manually - Ctrl+C on the text fields")
    
    with col4:
        if st.button("🗑️ Mark as delete", use_container_width=True):
            st.warning("Delete functionality: mark for removal (will implement in export)")
            # Could add a 'to_delete' column
    
    # Statistics
    st.divider()
    with st.expander("📊 Statistics & Progress"):
        edited_rows = len(set([log['row'] for log in st.session_state.edits_log]))
        st.metric("Rows edited", edited_rows)
        st.metric("Total edits", len(st.session_state.edits_log))
        
        if st.button("Show edit summary"):
            for log in st.session_state.edits_log[-5:]:  # Show last 5
                st.text(f"Row {log['row']}: Changed romanized from '{log['old_romanized'][:50]}...'")

else:
    # Welcome screen when no file loaded
    st.info("👈 **Get Started**: Upload your CSV file using the sidebar")
    
    st.markdown("""
    ### 📋 Instructions
    
    1. **Upload** your CSV file (must have `romanized` and `devanagari` columns)
    2. **Review** each sentence pair side-by-side
    3. **Edit** either column as needed
    4. **Save** changes (auto-advances to next row)
    5. **Download** corrected CSV when done
    
    ### ⚠️ Common Issues to Watch For
    
    - **Wrong word mapping**: `paryawaran` → should be `पर्यावरण`, not `वातावरण`
    - **Missing spaces**: Romanized `timi sangai` but Devanagari `तिमीसंगै` (space vs no space)
    - **Length mismatch**: Very different lengths often indicate missing words
    
    ### 💡 Tips
    
    - Use the **Quick insert** buttons for Devanagari vowel signs
    - **Flag rows** that need special attention
    - The **similarity score** highlights problematic pairs
    - Export periodically to save progress
    """)

    # Example data preview
    with st.expander("📖 Show example of correct format"):
        st.code("""romanized,devanagari
maile tyo kitab padhera ma sanga kehi naya jankari aayo,मैले त्यो किताब पढेर मसँग केही नयाँ जानकारी आयो
timi sangai aaja khana khane kura gareko thiyena,तिमीसंगै आज खाना खाने कुरा गरेको थिएन""", language='csv')