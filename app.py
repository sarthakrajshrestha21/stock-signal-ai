import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from engine import RecursiveSignalEngine
import os

# Page config
st.set_page_config(
    page_title="🧠 Recursive Stock AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .signal-strong-buy {
        color: #00c853;
        font-weight: bold;
    }
    .signal-buy {
        color: #2979ff;
        font-weight: bold;
    }
    .signal-watch {
        color: #ffc107;
        font-weight: bold;
    }
    .signal-avoid {
        color: #ff1744;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'engine' not in st.session_state:
    st.session_state.engine = RecursiveSignalEngine()
if 'df' not in st.session_state:
    st.session_state.df = None
if 'signals' not in st.session_state:
    st.session_state.signals = None
if 'trained' not in st.session_state:
    st.session_state.trained = False

# Sidebar
with st.sidebar:
    st.title("🧠 Recursive AI")
    st.markdown("---")

    st.markdown("### 📁 Upload Data")
    uploaded_file = st.file_uploader(
        "Upload Excel file (sheets = YYYY_MM_DD)",
        type=['xlsx', 'xls'],
        help="Each sheet should be named like 2024_01_15 and contain stock data"
    )

    st.markdown("---")
    st.markdown("### 🔄 Recursive Learning")

    if st.session_state.engine.is_fitted:
        st.success("✅ Model Loaded")
        history = st.session_state.engine.get_training_history()
        if history:
            st.info(f"Trained {len(history)} time(s)")
            st.caption(f"Last: {history[-1]['timestamp'][:10]}")
    else:
        st.warning("⚠️ No model yet")

    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    if st.session_state.df is not None:
        df = st.session_state.df
        st.metric("Total Rows", f"{len(df):,}")
        st.metric("Symbols", df['Symbol'].nunique())
        st.metric("Date Range", f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        st.metric("Unique Dates", df['Date'].nunique())

# Main content
st.markdown('<p class="main-header">🧠 Recursive Stock Signal Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload your stock data → AI recursively learns → Get trade signals</p>', unsafe_allow_html=True)

st.markdown("---")

# Process uploaded file
if uploaded_file is not None:
    try:
        with st.spinner("📖 Parsing multi-sheet Excel..."):
            df = st.session_state.engine.parse_excel(uploaded_file)
            st.session_state.df = df
        st.success(f"✅ Loaded {len(df):,} rows | {df['Symbol'].nunique()} symbols | {df['Date'].nunique()} dates")
    except Exception as e:
        st.error(f"❌ Error parsing file: {str(e)}")
        st.info("💡 Make sure sheet names are in YYYY_MM_DD format (e.g., 2024_01_15)")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Explorer", "🧠 Train AI", "🎯 Signals", "📈 Analytics"])

# TAB 1: Data Explorer
with tab1:
    if st.session_state.df is not None:
        df = st.session_state.df

        col1, col2 = st.columns([1, 3])

        with col1:
            st.subheader("🔍 Filters")
            selected_symbol = st.selectbox("Select Symbol", sorted(df['Symbol'].unique()))
            selected_date = st.date_input("Jump to Date", value=df['Date'].max())

            st.markdown("---")
            st.subheader("📋 Latest Data")
            latest_data = df[df['Symbol'] == selected_symbol].tail(1)
            if not latest_data.empty:
                l = latest_data.iloc[0]
                st.metric("Close", f"{l['Close']:.2f}")
                st.metric("VWAP", f"{l['VWAP']:.2f}")
                st.metric("Volume", f"{l['Vol']:,.0f}")
                st.metric("Range %", f"{l['Range %']:.2f}%")

        with col2:
            # Price chart
            sym_df = df[df['Symbol'] == selected_symbol].sort_values('Date')

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.03, row_heights=[0.7, 0.3])

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=sym_df['Date'],
                open=sym_df['Open'],
                high=sym_df['High'],
                low=sym_df['Low'],
                close=sym_df['Close'],
                name="Price"
            ), row=1, col=1)

            # Moving averages
            fig.add_trace(go.Scatter(x=sym_df['Date'], y=sym_df['120 Days'], 
                                    name="120 MA", line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=sym_df['Date'], y=sym_df['180 Days'], 
                                    name="180 MA", line=dict(color='purple')), row=1, col=1)

            # Volume
            fig.add_trace(go.Bar(x=sym_df['Date'], y=sym_df['Vol'], name="Volume", 
                                marker_color='blue'), row=2, col=1)

            fig.update_layout(title=f"{selected_symbol} - Price & Volume", 
                             height=600, showlegend=True)
            fig.update_xaxes(rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            # Raw data table
            with st.expander("📋 View Raw Data"):
                st.dataframe(sym_df.tail(50), use_container_width=True)
    else:
        st.info("📁 Upload an Excel file to explore data")

# TAB 2: Train AI
with tab2:
    if st.session_state.df is not None:
        df = st.session_state.df

        st.subheader("🧠 Recursive Model Training")
        st.markdown("""
        The AI will:
        1. **Engineer 20+ technical features** from your raw data
        2. **Validate** using time-series cross-validation (no data leakage)
        3. **Train** the final model on all data
        4. **Save** the model for future recursive learning
        """)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 START RECURSIVE TRAINING", type="primary", use_container_width=True):
                with st.spinner("🧠 Engineering features & training model..."):
                    try:
                        validation_results, features = st.session_state.engine.train(df, validate=True)
                        st.session_state.trained = True

                        st.balloons()
                        st.success("✅ Model trained and saved for recursive learning!")

                        if validation_results:
                            st.markdown("### 📊 Validation Results (Time-Series CV)")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Accuracy", f"{validation_results['accuracy']:.1%}")
                            c2.metric("Precision", f"{validation_results['precision']:.1%}")
                            c3.metric("Recall", f"{validation_results['recall']:.1%}")

                            st.line_chart({
                                'Fold Accuracy': validation_results['fold_accuracies']
                            })
                    except Exception as e:
                        st.error(f"❌ Training failed: {str(e)}")

        with col2:
            st.markdown("### 🧬 What the AI Learns")
            st.markdown("""
            - **Momentum**: Returns over 1d, 5d, 10d
            - **Volatility**: 5d & 20d rolling std
            - **Trend**: MA ratios, distance from MAs
            - **Position**: Where price sits in 52w range
            - **Volume**: Volume spikes vs averages
            - **VWAP**: Deviation from fair value
            - **Candlestick**: Body size, shadows
            - **Efficiency**: Turnover per transaction
            """)

        # Feature importance
        if st.session_state.engine.is_fitted:
            st.markdown("---")
            st.subheader("🎯 Feature Importance (What the AI Values Most)")
            importance_df = st.session_state.engine.get_feature_importance()
            if importance_df is not None:
                fig = px.bar(importance_df.head(15), x='importance', y='feature', 
                            orientation='h', color='importance',
                            color_continuous_scale='Viridis')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("📊 Full Feature Ranking"):
                    st.dataframe(importance_df, use_container_width=True)
    else:
        st.info("📁 Upload data first, then train the AI")

# TAB 3: Signals
with tab3:
    if st.session_state.df is not None and st.session_state.engine.is_fitted:
        st.subheader("🎯 Today's Trade Signals")
        st.markdown("Signals are ranked by AI confidence (probability of >1.5% gain in 5 days)")

        try:
            signals = st.session_state.engine.predict_signals(st.session_state.df)
            st.session_state.signals = signals

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            strong_buy = len(signals[signals['recommendation'] == '🔥 STRONG_BUY'])
            buy = len(signals[signals['recommendation'] == '🟢 BUY'])
            watch = len(signals[signals['recommendation'] == '🟡 WATCH'])
            avoid = len(signals[signals['recommendation'].isin(['🟠 WEAK', '🔴 AVOID'])])

            col1.metric("🔥 Strong Buy", strong_buy)
            col2.metric("🟢 Buy", buy)
            col3.metric("🟡 Watch", watch)
            col4.metric("🔴 Avoid/Weak", avoid)

            # Signals table
            st.markdown("---")

            # Color-coded display
            def color_recommendation(val):
                if 'STRONG_BUY' in val:
                    return 'background-color: #00c853; color: white; font-weight: bold'
                elif 'BUY' in val and 'STRONG' not in val:
                    return 'background-color: #2979ff; color: white; font-weight: bold'
                elif 'WATCH' in val:
                    return 'background-color: #ffc107; color: black; font-weight: bold'
                elif 'AVOID' in val:
                    return 'background-color: #ff1744; color: white; font-weight: bold'
                return ''

            styled_signals = signals.style.applymap(color_recommendation, subset=['recommendation'])
            st.dataframe(styled_signals, use_container_width=True, height=500)

            # Download signals
            csv = signals.to_csv(index=False)
            st.download_button(
                label="📥 Download Signals as CSV",
                data=csv,
                file_name=f"signals_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

            # Strong buys highlight
            strong_buys = signals[signals['recommendation'] == '🔥 STRONG_BUY']
            if not strong_buys.empty:
                st.markdown("---")
                st.subheader("🚀 STRONG BUY Signals")
                st.write("These have >75% AI confidence")

                for _, row in strong_buys.head(10).iterrows():
                    with st.container():
                        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
                        c1.markdown(f"**{row['Symbol']}** — {row['Date'].strftime('%Y-%m-%d')}")
                        c2.metric("Price", f"{row['current_price']:.2f}")
                        c3.metric("Confidence", f"{row['confidence']:.1%}")
                        c4.metric("Volume", f"{row['Vol']:,.0f}")

        except Exception as e:
            st.error(f"❌ Signal generation failed: {str(e)}")
    else:
        if st.session_state.df is None:
            st.info("📁 Step 1: Upload your Excel file")
        else:
            st.info("🧠 Step 2: Go to 'Train AI' tab and train the model")

# TAB 4: Analytics
with tab4:
    if st.session_state.df is not None:
        df = st.session_state.df

        st.subheader("📈 Market Analytics")

        # Market overview
        latest_date = df['Date'].max()
        latest_df = df[df['Date'] == latest_date]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 📊 Top Gainers (Latest Day)")
            latest_df['day_return'] = (latest_df['Close'] - latest_df['Open']) / latest_df['Open']
            gainers = latest_df.nlargest(10, 'day_return')[['Symbol', 'day_return', 'Range %', 'Vol']]
            gainers['day_return'] = gainers['day_return'].apply(lambda x: f"{x:.2%}")
            st.dataframe(gainers, use_container_width=True)

        with col2:
            st.markdown("### 📉 Top Losers (Latest Day)")
            losers = latest_df.nsmallest(10, 'day_return')[['Symbol', 'day_return', 'Range %', 'Vol']]
            losers['day_return'] = losers['day_return'].apply(lambda x: f"{x:.2%}")
            st.dataframe(losers, use_container_width=True)

        with col3:
            st.markdown("### 🔊 Most Active (Volume)")
            active = latest_df.nlargest(10, 'Vol')[['Symbol', 'Vol', 'Turnover', 'Trans']]
            st.dataframe(active, use_container_width=True)

        # Distribution charts
        st.markdown("---")
        st.subheader("📊 Market Distributions")

        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(latest_df, x='Range %', nbins=50, 
                              title="Intraday Range % Distribution",
                              color_discrete_sequence=['#667eea'])
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.scatter(latest_df, x='VWAP %', y='Range %', 
                            size='Vol', color='Trans',
                            title="VWAP Deviation vs Range",
                            hover_data=['Symbol'])
            st.plotly_chart(fig, use_container_width=True)

        # 52-week analysis
        st.markdown("---")
        st.subheader("🏔️ 52-Week Range Analysis")
        latest_df['range_pct'] = (latest_df['Close'] - latest_df['52 Weeks Low']) / (latest_df['52 Weeks High'] - latest_df['52 Weeks Low'])

        fig = px.histogram(latest_df, x='range_pct', nbins=50,
                          title="Position in 52-Week Range (0 = 52w Low, 1 = 52w High)",
                          color_discrete_sequence=['#764ba2'])
        fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Midpoint")
        st.plotly_chart(fig, use_container_width=True)

        # Training history
        if st.session_state.engine.is_fitted:
            history = st.session_state.engine.get_training_history()
            if history:
                st.markdown("---")
                st.subheader("🧠 Recursive Learning History")
                hist_df = pd.DataFrame(history)
                st.dataframe(hist_df, use_container_width=True)

                if len(history) > 1 and all('validation' in h and h['validation'] for h in history):
                    accs = [h['validation']['accuracy'] for h in history]
                    fig = px.line(x=range(1, len(accs)+1), y=accs, 
                                 labels={'x': 'Training Iteration', 'y': 'Accuracy'},
                                 title="Recursive Learning Progress",
                                 markers=True)
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📁 Upload data to see analytics")

# Footer
st.markdown("---")
st.caption("🧠 Recursive Stock AI | Built with Streamlit + scikit-learn | 100% Free | Vibe Coded")
