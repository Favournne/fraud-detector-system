# investigation.py
import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Account Investigation", layout="wide")

# ---------- Sidebar navigation ----------
st.sidebar.title("Navigation")
st.sidebar.markdown("[Live Monitor](http://localhost:8501)")
st.sidebar.markdown("[Account Investigation](http://localhost:8502)")
st.sidebar.divider()
st.sidebar.caption("Data from FastAPI (port 8000)")

# ---------- Main content ----------
st.title(" Account Investigation Dashboard")
st.caption("Search for a nameOrig and see transaction history with explanations (no auto-refresh)")

with st.form(key="investigation_form"):
    col1, col2 = st.columns([3, 1])
    with col1:
        name_orig = st.text_input(
            "Enter nameOrig (e.g., C123456789)",
            value="C123456789"
        )
    with col2:
        limit = st.number_input("Number of transactions", min_value=1, max_value=50, value=10)
    submitted = st.form_submit_button("Fetch Transactions", type="primary")

if submitted:
    if not name_orig.strip():
        st.warning("Please enter a nameOrig.")
        st.stop()

    with st.spinner(f"Fetching data for {name_orig}..."):
        try:
            stats_resp = requests.get(
                f"http://127.0.0.1:8000/v1/account/{name_orig}/stats",
                timeout=5
            )
            tx_resp = requests.get(
                f"http://127.0.0.1:8000/v1/account/{name_orig}/transactions?limit={limit}",
                timeout=5
            )

            if stats_resp.status_code != 200 or tx_resp.status_code != 200:
                st.error("Failed to fetch data. Check FastAPI and account ID.")
                st.stop()

            stats = stats_resp.json()
            data = tx_resp.json()
            df = pd.DataFrame(data['transactions'])

            st.markdown("---")
            st.subheader(f"Account Summary: `{name_orig}`")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Transactions", f"{stats['total_transactions']:,}")
            col2.metric("Blocked", f"{stats['blocked_count']:,}")
            col3.metric("Total Amount", f"₦{stats['total_amount']:,.2f}")
            col4.metric("Avg Amount", f"₦{stats['avg_amount']:,.2f}")

            st.markdown("---")
            st.subheader(f"Last {len(df)} Transactions")
            if df.empty:
                st.info("No transactions found.")
            else:
                display_df = df.copy()
                display_df['amount'] = display_df['amount'].apply(lambda x: f"₦{float(x):,.2f}")
                display_df['fraud_probability'] = display_df['fraud_probability'].apply(lambda x: f"{float(x):.2%}")
                st.dataframe(
                    display_df[['transaction_id', 'amount', 'action', 'fraud_probability', 'created_at', 'reason']],
                    use_container_width=True,
                    hide_index=True
                )

                blocked = df[df['action'] == 'BLOCKED']
                if not blocked.empty:
                    st.warning(f"{len(blocked)} blocked transaction(s) found. Check the 'reason' column.")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to FastAPI. Make sure it's running on port 8000.")
        except Exception as e:
            st.error(f"An error occurred: {e}")