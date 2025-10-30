"""
Streamlit Dashboard for Live Trading Simulator
Real-time visualization of portfolio, signals, and trades
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from database import DatabaseManager
import config

# Page config
st.set_page_config(
    page_title="Live Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
@st.cache_resource
def get_database():
    return DatabaseManager()

db = get_database()


def format_currency(value):
    """Format as currency"""
    return f"${value:,.2f}"


def format_percent(value):
    """Format as percentage"""
    return f"{value:.2f}%"


def create_portfolio_chart(df):
    """Create portfolio value over time chart"""
    if len(df) == 0:
        return None

    fig = go.Figure()

    # Portfolio value line
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['portfolio_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#00CC96', width=2),
        fill='tonexty',
        fillcolor='rgba(0, 204, 150, 0.1)'
    ))

    # Add initial capital reference line
    fig.add_hline(
        y=config.INITIAL_CAPITAL,
        line_dash="dash",
        line_color="gray",
        annotation_text="Initial Capital"
    )

    fig.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        hovermode='x unified',
        height=400
    )

    return fig


def create_returns_chart(df):
    """Create returns chart"""
    if len(df) == 0:
        return None

    fig = go.Figure()

    # Returns line
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['return_pct'],
        mode='lines',
        name='Return %',
        line=dict(color='#636EFA', width=2)
    ))

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title="Cumulative Returns",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        hovermode='x unified',
        height=400
    )

    return fig


def create_signals_chart(signals_df):
    """Create signals distribution chart"""
    if len(signals_df) == 0:
        return None

    # Count signals by action
    action_counts = signals_df['action'].value_counts()

    fig = px.bar(
        x=action_counts.index,
        y=action_counts.values,
        labels={'x': 'Action', 'y': 'Count'},
        title='Signal Distribution',
        color=action_counts.index,
        color_discrete_map={
            'BUY': '#00CC96',
            'SELL': '#EF553B',
            'HOLD': '#636EFA',
            'CLOSE': '#FFA15A',
            'CLOSE_ALL': '#AB63FA'
        }
    )

    fig.update_layout(height=300, showlegend=False)

    return fig


def create_regime_chart(signals_df):
    """Create regime distribution chart"""
    if len(signals_df) == 0:
        return None

    # Count regimes
    regime_counts = signals_df['extreme_condition'].value_counts()

    fig = px.pie(
        values=regime_counts.values,
        names=regime_counts.index,
        title='Market Regime Distribution',
        color_discrete_map={
            'NORMAL': '#00CC96',
            'EXTREME_BEAR': '#FFA15A',
            'CRISIS': '#EF553B',
            'ERROR': '#AB63FA',
            'UNKNOWN': '#B6E880'
        }
    )

    fig.update_layout(height=300)

    return fig


def main():
    """Main dashboard"""

    # Title
    st.title("📈 Live Trading Dashboard")
    st.markdown("---")

    # Refresh button
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("🔄 Refresh Data"):
            st.cache_resource.clear()
            st.rerun()

    # Get current position
    current_position = db.get_current_position()

    if current_position is None:
        st.warning("⚠️ No portfolio data yet. Run the simulator first!")
        st.code("python run_live_simulator.py", language="bash")
        return

    # Current Status Cards
    st.subheader("📊 Current Status")

    col1, col2, col3, col4 = st.columns(4)

    portfolio_value = current_position['portfolio_value']
    cash = current_position['cash']
    position_value = current_position['position_value']
    return_pct = current_position['return_pct'] or 0
    spy_quantity = current_position['spy_quantity'] or 0

    with col1:
        st.metric(
            "Portfolio Value",
            format_currency(portfolio_value),
            format_percent(return_pct),
            delta_color="normal"
        )

    with col2:
        st.metric(
            "Cash",
            format_currency(cash),
            None
        )

    with col3:
        st.metric(
            "Position Value",
            format_currency(position_value),
            None
        )

    with col4:
        st.metric(
            "SPY Shares",
            f"{spy_quantity:,}",
            None
        )

    st.markdown("---")

    # Portfolio History
    st.subheader("📈 Portfolio Performance")

    # Date range selector
    days_options = [7, 14, 30, 60, 90, 180, 365]
    selected_days = st.selectbox(
        "Select time period:",
        days_options,
        index=2  # Default 30 days
    )

    portfolio_history = db.get_portfolio_history(days=selected_days)

    if len(portfolio_history) > 0:
        col1, col2 = st.columns(2)

        with col1:
            chart = create_portfolio_chart(portfolio_history)
            if chart:
                st.plotly_chart(chart, use_container_width=True)

        with col2:
            chart = create_returns_chart(portfolio_history)
            if chart:
                st.plotly_chart(chart, use_container_width=True)

    else:
        st.info("No portfolio history available yet.")

    st.markdown("---")

    # Recent Trades
    st.subheader("💼 Recent Trades")

    trades_df = db.get_recent_trades(limit=10)

    if len(trades_df) > 0:
        # Format for display
        trades_display = trades_df[['timestamp', 'side', 'quantity', 'price', 'total_cost', 'reason']].copy()
        trades_display['timestamp'] = pd.to_datetime(trades_display['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        trades_display['price'] = trades_display['price'].apply(lambda x: f"${x:.2f}")
        trades_display['total_cost'] = trades_display['total_cost'].apply(lambda x: f"${x:,.2f}")

        st.dataframe(trades_display, use_container_width=True, hide_index=True)
    else:
        st.info("No trades executed yet.")

    st.markdown("---")

    # Recent Signals
    st.subheader("🎯 Recent Signals")

    signals_df = db.get_recent_signals(limit=20)

    if len(signals_df) > 0:
        col1, col2 = st.columns(2)

        with col1:
            chart = create_signals_chart(signals_df)
            if chart:
                st.plotly_chart(chart, use_container_width=True)

        with col2:
            chart = create_regime_chart(signals_df)
            if chart:
                st.plotly_chart(chart, use_container_width=True)

        # Signals table
        st.markdown("### Signal History")
        signals_display = signals_df[['timestamp', 'action', 'reason', 'prediction',
                                      'extreme_condition', 'position_size']].copy()
        signals_display['timestamp'] = pd.to_datetime(signals_display['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        signals_display['prediction'] = signals_display['prediction'].apply(lambda x: f"{x:.4f}")
        signals_display['position_size'] = signals_display['position_size'].apply(lambda x: f"{x*100:.1f}%")

        st.dataframe(signals_display, use_container_width=True, hide_index=True)
    else:
        st.info("No signals generated yet.")

    # Footer
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption("Data updates when simulator runs. Use 'Refresh Data' button to reload.")


if __name__ == "__main__":
    main()
