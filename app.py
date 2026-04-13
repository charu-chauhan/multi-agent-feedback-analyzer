import streamlit as st
import pandas as pd
import os
import shutil

import config
from pipeline import run_pipeline

st.set_page_config(page_title="Feedback Analysis System", layout="wide")
st.title("🎯 Intelligent User Feedback Analysis System")

# Ensure directories exist
os.makedirs(config.DATA_DIR, exist_ok=True)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# --- Sidebar: API Key + Configuration ---
st.sidebar.header("🔑 API Key")
api_key = st.sidebar.text_input("Gemini API Key", value=config.GEMINI_API_KEY, type="password")
if api_key:
    config.GEMINI_API_KEY = api_key
    os.environ["GEMINI_API_KEY"] = api_key

st.sidebar.header("⚙️ Classification Settings")
config.CLASSIFICATION_CONFIDENCE_THRESHOLD = st.sidebar.slider(
    "Confidence Threshold", 0.0, 1.0, config.CLASSIFICATION_CONFIDENCE_THRESHOLD, 0.05,
    help="Items below this threshold will be flagged as uncertain",
)

st.sidebar.subheader("Default Category Priorities")
for cat in config.VALID_CATEGORIES:
    config.CATEGORY_PRIORITY_DEFAULTS[cat] = st.sidebar.selectbox(
        f"{cat}", config.VALID_PRIORITIES,
        index=config.VALID_PRIORITIES.index(config.CATEGORY_PRIORITY_DEFAULTS[cat]),
        key=f"pri_{cat}",
    )

st.sidebar.subheader("Rating → Priority Mapping")
for rating in [1, 2, 3, 4, 5]:
    config.RATING_PRIORITY_MAP[rating] = st.sidebar.selectbox(
        f"Rating {rating} ⭐", config.VALID_PRIORITIES,
        index=config.VALID_PRIORITIES.index(config.RATING_PRIORITY_MAP[rating]),
        key=f"rat_{rating}",
    )

# --- Tabs ---
tab_upload, tab_process, tab_override, tab_analytics = st.tabs(
    ["📁 Upload Data", "▶️ Process Feedback", "✏️ Review & Override", "📈 Analytics"]
)

# --- Upload Tab ---
with tab_upload:
    st.subheader("Upload Input CSV Files")
    st.info("Upload your feedback data files. If not uploaded, the system uses the default sample data in the `data/` folder.")

    col1, col2, col3 = st.columns(3)
    with col1:
        reviews_file = st.file_uploader("App Store Reviews CSV", type="csv", key="reviews")
        if reviews_file:
            df = pd.read_csv(reviews_file)
            df.to_csv(config.APP_REVIEWS_CSV, index=False)
            st.success(f"✅ Uploaded {len(df)} reviews")
            st.dataframe(df, height=300)
        elif os.path.exists(config.APP_REVIEWS_CSV):
            df = pd.read_csv(config.APP_REVIEWS_CSV)
            st.caption(f"Using existing file ({len(df)} reviews)")
            st.dataframe(df, height=300)

    with col2:
        emails_file = st.file_uploader("Support Emails CSV", type="csv", key="emails")
        if emails_file:
            df = pd.read_csv(emails_file)
            df.to_csv(config.SUPPORT_EMAILS_CSV, index=False)
            st.success(f"✅ Uploaded {len(df)} emails")
            st.dataframe(df, height=300)
        elif os.path.exists(config.SUPPORT_EMAILS_CSV):
            df = pd.read_csv(config.SUPPORT_EMAILS_CSV)
            st.caption(f"Using existing file ({len(df)} emails)")
            st.dataframe(df, height=300)

    with col3:
        expected_file = st.file_uploader("Expected Classifications CSV", type="csv", key="expected")
        if expected_file:
            df = pd.read_csv(expected_file)
            df.to_csv(config.EXPECTED_CSV, index=False)
            st.success(f"✅ Uploaded {len(df)} classifications")
            st.dataframe(df, height=300)
        elif os.path.exists(config.EXPECTED_CSV):
            df = pd.read_csv(config.EXPECTED_CSV)
            st.caption(f"Using existing file ({len(df)} classifications)")
            st.dataframe(df, height=300)

# --- Process Tab ---
with tab_process:
    st.subheader("Run Feedback Processing Pipeline")

    if not config.GEMINI_API_KEY:
        st.warning("⚠️ Enter your Gemini API Key in the sidebar before running.")
    else:
        st.markdown("**Current Settings:**")
        col1, col2 = st.columns(2)
        col1.markdown(f"- Model: `{config.LLM_MODEL}`")
        col1.markdown(f"- Confidence Threshold: `{config.CLASSIFICATION_CONFIDENCE_THRESHOLD}`")
        col2.markdown("- Category Priorities: " + ", ".join(f"`{k}: {v}`" for k, v in config.CATEGORY_PRIORITY_DEFAULTS.items()))

        if st.button("🚀 Run Pipeline", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(msg, pct):
                status_text.text(msg)
                progress_bar.progress(pct)

            with st.spinner("Running multi-agent pipeline..."):
                try:
                    tickets, logs, metrics = run_pipeline(progress_callback=update_progress)
                    st.success(f"✅ Processed {metrics['total_processed']} items in {metrics['processing_time_seconds']}s")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Category Accuracy", f"{metrics['category_accuracy']*100:.1f}%")
                    col2.metric("Priority Accuracy", f"{metrics['priority_accuracy']*100:.1f}%")
                    col3.metric("Tickets/min", f"{metrics['tickets_per_minute']:.1f}")
                except Exception as e:
                    st.error(f"Pipeline error: {e}")

# --- Review & Override Tab ---
with tab_override:
    st.subheader("Review & Override Generated Tickets")
    if os.path.exists(config.GENERATED_TICKETS_CSV):
        tickets_df = pd.read_csv(config.GENERATED_TICKETS_CSV)

        st.markdown("**Editable fields:** Title, Description, Category, Priority, Technical Details, Status")

        # Configure which columns are editable
        column_config = {
            "ticket_id": st.column_config.TextColumn("Ticket ID", disabled=True),
            "source_id": st.column_config.TextColumn("Source ID", disabled=True),
            "source_type": st.column_config.TextColumn("Source", disabled=True),
            "title": st.column_config.TextColumn("Title", width="large"),
            "description": st.column_config.TextColumn("Description", width="large"),
            "category": st.column_config.SelectboxColumn("Category", options=config.VALID_CATEGORIES),
            "priority": st.column_config.SelectboxColumn("Priority", options=config.VALID_PRIORITIES),
            "confidence": st.column_config.NumberColumn("Confidence", format="%.2f", disabled=True),
            "technical_details": st.column_config.TextColumn("Technical Details", width="large"),
            "status": st.column_config.SelectboxColumn("Status", options=["New", "Needs Review", "Acknowledged", "In Progress", "Resolved", "Rejected"]),
            "created_date": st.column_config.TextColumn("Created", disabled=True),
            "quality_score": st.column_config.NumberColumn("Quality", disabled=True),
            "quality_notes": st.column_config.TextColumn("Quality Notes", disabled=True),
        }

        edited_df = st.data_editor(
            tickets_df, use_container_width=True, num_rows="fixed",
            column_config=column_config, height=500,
        )

        # Summary metrics from edited data (updates live)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tickets", len(edited_df))
        if "category" in edited_df.columns:
            col2.metric("Bugs", len(edited_df[edited_df["category"] == "Bug"]))
            col3.metric("Feature Requests", len(edited_df[edited_df["category"] == "Feature Request"]))
            col4.metric("Complaints", len(edited_df[edited_df["category"] == "Complaint"]))

        if st.button("💾 Save Changes"):
            edited_df.to_csv(config.GENERATED_TICKETS_CSV, index=False)
            st.success("✅ Tickets saved!")
    else:
        st.info("No tickets generated yet. Go to the **Process Feedback** tab to run the pipeline.")

# --- Analytics Tab ---
with tab_analytics:
    st.subheader("Processing Metrics & Analytics")

    if os.path.exists(config.METRICS_CSV):
        metrics_df = pd.read_csv(config.METRICS_CSV)
        row = metrics_df.iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Processed", int(row.get("total_processed", 0)))
        col2.metric("Category Accuracy", f"{row.get('category_accuracy', 0)*100:.1f}%")
        col3.metric("Priority Accuracy", f"{row.get('priority_accuracy', 0)*100:.1f}%")
        col4.metric("Processing Time", f"{row.get('processing_time_seconds', 0):.1f}s")
    else:
        st.info("No metrics yet. Run the pipeline first.")

    if os.path.exists(config.GENERATED_TICKETS_CSV):
        tickets_df = pd.read_csv(config.GENERATED_TICKETS_CSV)
        col1, col2 = st.columns(2)
        if "category" in tickets_df.columns:
            with col1:
                st.subheader("Tickets by Category")
                st.bar_chart(tickets_df["category"].value_counts())
        if "priority" in tickets_df.columns:
            with col2:
                st.subheader("Tickets by Priority")
                st.bar_chart(tickets_df["priority"].value_counts())

    if os.path.exists(config.PROCESSING_LOG_CSV):
        st.subheader("Processing Log")
        st.dataframe(pd.read_csv(config.PROCESSING_LOG_CSV), use_container_width=True)
