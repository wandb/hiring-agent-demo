import streamlit as st
import weave
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import itertools
import time

def init_weave_client(project_name):
    """Initialize the Weave client with the given project name."""
    try:
        client = weave.init(project_name)
        # Add standard costs for common models
        # MODEL_COSTS = [
        #     # model name, prompt cost, completion cost
        #     ("gpt-4o-2024-05-13", 0.00003, 0.00006),
        #     ("gpt-4o-mini-2024-07-18", 0.00003, 0.00006),
        #     ("gemini/gemini-1.5-flash", 0.00000025, 0.0000005),
        #     ("gpt-4o-mini", 0.00003, 0.00006),
        #     ("gpt-4-turbo", 0.00003, 0.00006),
        #     ("claude-3-haiku-20240307", 0.00001, 0.00003),
        #     ("gpt-4o", 0.00003, 0.00006),
        # ]
        
        # for model, prompt_cost, completion_cost in MODEL_COSTS:
        #     try:
        #         client.add_cost(
        #             llm_id=model,
        #             prompt_token_cost=prompt_cost,
        #             completion_token_cost=completion_cost,
        #         )
        #     except Exception:
        #         pass  # Ignore errors related to already existing costs
        
        return client
    except Exception as e:
        st.error(f"Failed to initialize Weave client for project '{project_name}': {e}")
        return None

def fetch_calls(client, project_id, start_timestamp, limit=5000):
    """Fetch calls data from Weave using MongoDB-style queries."""
    try:
        # Use entity and project from project_id
        entity, project = project_id.split('/')
        
        # Format the op_names pattern exactly as in the example
        op_name_pattern = f"weave:///{project_id}/op/HiringAgent.predict:*"
        
        # Construct the query with the correct timestamp format
        query = {
            "$expr": {
                "$gt": [
                    {"$getField": "started_at"}, 
                    {"$literal": start_timestamp}
                ]
            }
        }
        
        # Debug message to show what we're querying
        st.info(f"Querying for HiringAgent.predict calls since timestamp {start_timestamp} ({datetime.fromtimestamp(start_timestamp)})")
        
        # Execute the query
        calls = client.get_calls(
            filter={"op_names": [op_name_pattern]},
            query=query,
            sort_by=[{"field": "started_at", "direction": "desc"}],
            include_costs=True,
            include_feedback=True,
            limit=limit
        )
        
        st.success(f"Fetched {len(calls)} HiringAgent.predict calls since {datetime.fromtimestamp(start_timestamp)}")
        return calls
    except Exception as e:
        st.error(f"Error fetching calls: {e}")
        return []

def process_calls(calls):
    """Process the calls data and return a DataFrame."""
    records = []
    total_cost = 0
    
    for call in calls:
        feedback = call.summary.get("weave", {}).get("feedback", [])
        
        # Extract different feedback types
        thumbs_up = sum(1 for item in feedback if isinstance(item, dict) and item.get("payload", {}).get("emoji") == "👍")
        thumbs_down = sum(1 for item in feedback if isinstance(item, dict) and item.get("payload", {}).get("emoji") == "👎")
        expert_review = sum(1 for item in feedback if isinstance(item, dict) and "expert_review" in item.get("feedback_type", ""))
        robot_feedback = sum(1 for item in feedback if isinstance(item, dict) and item.get("payload", {}).get("emoji") == "🤖")
        
        # Extract cost information
        call_cost = 0
        if hasattr(call, 'summary') and call.summary and "weave" in call.summary:
            weave_summary = call.summary["weave"]
            
            # Try to get cost from the costs field
            if weave_summary and "costs" in weave_summary:
                for model_cost in weave_summary["costs"].values():
                    call_cost += model_cost.get("prompt_tokens_total_cost", 0)
                    call_cost += model_cost.get("completion_tokens_total_cost", 0)
            
            # Fallback to cost.total field
            elif weave_summary and "cost" in weave_summary and "total" in weave_summary["cost"]:
                call_cost = weave_summary["cost"]["total"]
        
        latency = call.summary.get("weave", {}).get("latency_ms", 0)
        
        records.append({
            "Call ID": call.id,
            "Trace ID": call.trace_id,
            "Display Name": call.display_name,
            "Latency (ms)": latency,
            "Thumbs Up": thumbs_up,
            "Thumbs Down": thumbs_down,
            "Expert Review": expert_review,
            "Robot Feedback": robot_feedback,
            "Started At": pd.to_datetime(getattr(call, "started_at", datetime.min)),
            "Cost": call_cost,
        })
        
        # Accumulate total cost
        total_cost += call_cost
    
    df = pd.DataFrame(records)
    return df, total_cost

def query_costs(client, limit=5000):
    """Query costs from Weave API with limit parameter."""
    try:
        costs = client.query_costs(limit=limit)
        if costs:
            df_costs = pd.DataFrame([cost.dict() for cost in costs])
            df_costs["total_cost"] = df_costs["prompt_token_cost"] + df_costs["completion_token_cost"]
            return df_costs
        else:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def plot_feedback_summary(df):
    """Create a pie chart showing the distribution of feedback types."""
    # Count total calls with each feedback type
    thumbs_up = df["Thumbs Up"].sum()
    thumbs_down = df["Thumbs Down"].sum()
    expert_review = df["Expert Review"].sum() 
    robot_feedback = df["Robot Feedback"].sum()
    
    # Calculate calls with no feedback
    # A call has no feedback if all feedback types are zero
    no_feedback_count = len(df[
        (df["Thumbs Up"] == 0) & 
        (df["Thumbs Down"] == 0) & 
        (df["Expert Review"] == 0) & 
        (df["Robot Feedback"] == 0)
    ])
    
    feedback_counts = {
        "Thumbs Up": thumbs_up,
        "Thumbs Down": thumbs_down,
        "Expert Review": expert_review,
        "Robot Feedback": robot_feedback,
        "No Feedback": no_feedback_count
    }
    
    # Remove types with zero count
    feedback_counts = {k: v for k, v in feedback_counts.items() if v > 0}
    
    if sum(feedback_counts.values()) == 0:
        return go.Figure().update_layout(title="No feedback data available")
    
    # Set custom colors, including a gray color for No Feedback
    colors = ["#66b3ff", "#ff9999", "#ffcc99", "#99ff99", "#cccccc"]
    
    fig = go.Figure(
        data=[
            go.Pie(
                labels=list(feedback_counts.keys()),
                values=list(feedback_counts.values()),
                marker={"colors": colors},
                hole=0.3,
            )
        ]
    )
    fig.update_traces(textinfo="percent+label", hoverinfo="label+value")
    fig.update_layout(title="Feedback Distribution")
    return fig

def plot_requests_over_time(df):
    """Create a line chart showing requests over time."""
    if df.empty or "Started At" not in df:
        return go.Figure().update_layout(title="No timestamp data available")
    
    # Convert to datetime and create hourly bins
    df["Hour"] = df["Started At"].dt.floor("H")
    
    # Count requests per hour
    requests_per_hour = df.groupby("Hour").size().reset_index()
    requests_per_hour.columns = ["Hour", "Count"]
    
    # Create line chart
    fig = px.line(
        requests_per_hour,
        x="Hour",
        y="Count",
        title="Requests Over Time",
        markers=True,
    )
    fig.update_layout(xaxis_title="Time", yaxis_title="Number of Requests")
    return fig

def plot_latency_distribution(df):
    """Create a histogram showing the distribution of latency."""
    if df.empty or "Latency (ms)" not in df or df["Latency (ms)"].isna().all():
        return go.Figure().update_layout(title="No latency data available")
    
    # Create histogram
    fig = px.histogram(
        df,
        x="Latency (ms)",
        title="Latency Distribution",
        nbins=20,
    )
    fig.update_layout(xaxis_title="Latency (ms)", yaxis_title="Count")
    return fig

def plot_cost_by_model(df_costs):
    """Create a bar chart showing cost by model."""
    if df_costs.empty or "llm_id" not in df_costs:
        return go.Figure().update_layout(title="No cost data available")
    
    # Create bar chart
    fig = px.bar(
        df_costs,
        x="llm_id",
        y="total_cost",
        title="Cost by Model",
        color="llm_id",
    )
    fig.update_layout(xaxis_title="Model", yaxis_title="Cost (USD)")
    return fig

def render_monitoring_dashboard():
    """Render the monitoring dashboard in Streamlit."""
    
    # Use project name directly from session state
    project_name = f"{st.session_state.wandb_entity}/{st.session_state.wandb_project}"
    
    # Display dashboard configuration at the top
    st.write(f"**Project:** {project_name}")
    
    # Add time range control in the main area
    col1, col2 = st.columns([3, 1])
    with col1:
        time_range = st.selectbox(
            "Time Range",
            ["Last 24 hours", "Last 7 days", "Last 30 days", "All time"],
            index=2,  # Default to 30 days
        )
        
        # Calculate timestamp for each range
        current_time = time.time()
        time_ranges = {
            "Last 24 hours": current_time - (24 * 60 * 60),
            "Last 7 days": current_time - (7 * 24 * 60 * 60),
            "Last 30 days": current_time - (30 * 24 * 60 * 60),
            "All time": 0,  # Beginning of time
        }
        start_timestamp = time_ranges[time_range]
    
    with col2:
        refresh = st.button("Refresh", use_container_width=True)
        auto_refresh = st.checkbox("Auto-refresh")
    
    # Add a divider
    st.divider()
    
    # Initialize Weave client
    client = init_weave_client(project_name)
    
    if not client:
        st.error("Failed to initialize Weave client. Please check your project name and try again.")
        return
    
    # Fetch data - fixed data limit at 5000 to ensure we get plenty of results
    with st.spinner("Fetching data from Weave..."):
        calls = fetch_calls(client, project_name, start_timestamp)
        df_calls, total_cost_from_calls = process_calls(calls)
        df_costs = query_costs(client)
    
    # Create dashboard sections
    st.subheader("HiringAgent Overview")
    
    # Calculate total cost
    total_cost = max(total_cost_from_calls, df_costs["total_cost"].sum() if not df_costs.empty else 0)
    
    # Overview metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Calls", len(df_calls))
    with col2:
        st.metric("Expert Reviews", df_calls["Expert Review"].sum())
    with col3:
        st.metric("Avg. Latency (ms)", int(df_calls["Latency (ms)"].mean()) if not df_calls.empty else 0)
    with col4:
        st.metric("Total Cost ($)", f"{total_cost:.4f}")
    
    # Cost analysis - moved up as requested
    st.subheader("Cost Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_cost_by_model(df_costs), use_container_width=True)
    with col2:
        if not df_costs.empty:
            cost_metrics = df_costs.groupby("llm_id").agg({
                "prompt_token_cost": "sum",
                "completion_token_cost": "sum",
                "total_cost": "sum"
            }).reset_index()
            cost_metrics = cost_metrics.round(6)
            st.dataframe(cost_metrics, hide_index=True)
        else:
            st.info("No cost data available")
    
    # Feedback distribution
    st.subheader("Feedback Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_feedback_summary(df_calls), use_container_width=True)
    with col2:
        # Calculate calls with no feedback
        no_feedback_count = len(df_calls[
            (df_calls["Thumbs Up"] == 0) & 
            (df_calls["Thumbs Down"] == 0) & 
            (df_calls["Expert Review"] == 0) & 
            (df_calls["Robot Feedback"] == 0)
        ])
        
        feedback_metrics = pd.DataFrame({
            "Feedback Type": ["Thumbs Up", "Thumbs Down", "Expert Review", "Robot Feedback", "No Feedback"],
            "Count": [
                df_calls["Thumbs Up"].sum(),
                df_calls["Thumbs Down"].sum(),
                df_calls["Expert Review"].sum(),
                df_calls["Robot Feedback"].sum(),
                no_feedback_count
            ]
        })
        st.dataframe(feedback_metrics, hide_index=True)
    
    # Requests over time
    st.subheader("Traffic Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_requests_over_time(df_calls), use_container_width=True)
    with col2:
        st.plotly_chart(plot_latency_distribution(df_calls), use_container_width=True)
    
    # Recent calls table
    st.subheader("Recent Calls")
    if not df_calls.empty:
        # Show the most recent calls with key information
        recent_calls = df_calls[["Call ID", "Display Name", "Started At", "Latency (ms)", "Cost"]].head(10)
        st.dataframe(recent_calls, hide_index=True)
    else:
        st.info("No recent calls data available")
    
    # Set up auto-refresh if enabled
    if auto_refresh:
        st.info("Auto-refreshing every 60 seconds")
        st.experimental_rerun()

if __name__ == "__main__":
    render_monitoring_dashboard() 