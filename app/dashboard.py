"""
CommissionIQ Dashboard
Interactive web interface for commissioning delay predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add the models directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
models_dir = os.path.join(parent_dir, 'models')
sys.path.insert(0, models_dir)

from train_model import CommissionDelayPredictor

# Page configuration
st.set_page_config(
    page_title="CommissionIQ - Predictive Analytics",
    page_icon="🏗️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .big-metric {
        font-size: 36px;
        font-weight: bold;
        color: #1f77b4;
    }
    .risk-critical {
        background-color: #ff4444;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .risk-high {
        background-color: #ff9800;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #ffeb3b;
        color: black;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .risk-low {
        background-color: #4caf50;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Remove caching to avoid compatibility issues
def load_model():
    """Load the trained model"""
    predictor = CommissionDelayPredictor()
    predictor.load_model()
    return predictor

def load_data():
    """Load commissioning data"""
    # Get paths relative to the app directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(parent_dir, 'data')
    
    projects = pd.read_csv(os.path.join(data_dir, 'projects.csv'))
    activities = pd.read_csv(os.path.join(data_dir, 'commissioning_activities.csv'))
    return projects, activities

def main():
    # Header
    st.title("🏗️ CommissionIQ - AI-Powered Delay Prediction")
    st.markdown("**Predict commissioning delays before they happen | Built for Facility Grid**")
    st.markdown("---")
    
    # Load data and model
    with st.spinner("Loading model and data..."):
        predictor = load_model()
        projects_df, activities_df = load_data()
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    page = st.sidebar.radio("Navigation", [
        "🎯 Overview",
        "🔮 Predictions",
        "📈 Analytics",
        "⚙️ Single Activity Predictor"
    ])
    
    if page == "🎯 Overview":
        show_overview(activities_df, projects_df, predictor)
    elif page == "🔮 Predictions":
        show_predictions(activities_df, predictor)
    elif page == "📈 Analytics":
        show_analytics(activities_df, predictor)
    elif page == "⚙️ Single Activity Predictor":
        show_single_predictor(predictor)

def show_overview(activities_df, projects_df, predictor):
    """Show overview dashboard"""
    
    st.header("📊 Portfolio Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Projects", len(projects_df))
    with col2:
        st.metric("Total Activities", len(activities_df))
    with col3:
        delayed = activities_df['is_delayed'].sum()
        st.metric("Delayed Activities", f"{delayed:,}", 
                 delta=f"{delayed/len(activities_df)*100:.1f}%", 
                 delta_color="inverse")
    with col4:
        avg_delay = activities_df[activities_df['is_delayed']]['delay_days'].mean()
        st.metric("Avg Delay", f"{avg_delay:.1f} days", delta_color="off")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Delay Rate by System Type")
        delay_by_system = activities_df.groupby('system_type').agg({
            'is_delayed': ['sum', 'count', 'mean']
        }).round(3)
        delay_by_system.columns = ['Delayed', 'Total', 'Delay Rate']
        delay_by_system['Delay Rate'] = delay_by_system['Delay Rate'] * 100
        
        fig = px.bar(
            delay_by_system.reset_index(),
            x='system_type',
            y='Delay Rate',
            color='Delay Rate',
            color_continuous_scale='RdYlGn_r',
            labels={'system_type': 'System Type', 'Delay Rate': 'Delay Rate (%)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Complexity vs Delay Rate")
        complexity_delay = activities_df.groupby('complexity').agg({
            'is_delayed': ['mean', 'count']
        }).reset_index()
        complexity_delay.columns = ['Complexity', 'Delay Rate', 'Count']
        complexity_delay['Delay Rate'] = complexity_delay['Delay Rate'] * 100
        
        # Sort by complexity level
        complexity_order = ['Low', 'Medium', 'High', 'Critical']
        complexity_delay['Complexity'] = pd.Categorical(
            complexity_delay['Complexity'], 
            categories=complexity_order, 
            ordered=True
        )
        complexity_delay = complexity_delay.sort_values('Complexity')
        
        fig = px.bar(
            complexity_delay,
            x='Complexity',
            y='Delay Rate',
            color='Delay Rate',
            color_continuous_scale='Reds',
            text='Delay Rate',
            labels={'Delay Rate': 'Delay Rate (%)'}
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.markdown("---")
    st.subheader("🔍 Top Risk Factors (ML Model Insights)")
    
    feature_importance = predictor.feature_importance.head(10).copy()
    feature_importance['importance'] = feature_importance['importance'] * 100
    
    # Rename features for readability
    feature_map = {
        'document_completeness_pct': 'Document Completeness',
        'contractor_performance_score': 'Contractor Performance',
        'planned_duration_days': 'Planned Duration',
        'equipment_type_encoded': 'Equipment Type',
        'contractor_encoded': 'Contractor',
        'num_dependencies': 'Number of Dependencies',
        'activity_type_encoded': 'Activity Type',
        'resource_availability_encoded': 'Resource Availability',
        'system_type_encoded': 'System Type',
        'complexity_encoded': 'Complexity'
    }
    feature_importance['feature'] = feature_importance['feature'].map(feature_map)
    
    fig = px.bar(
        feature_importance,
        y='feature',
        x='importance',
        orientation='h',
        color='importance',
        color_continuous_scale='Blues',
        labels={'feature': 'Risk Factor', 'importance': 'Importance (%)'}
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def show_predictions(activities_df, predictor):
    """Show prediction results"""
    
    st.header("🔮 AI Delay Predictions")
    
    # Get predictions for in-progress activities
    in_progress = activities_df[
        activities_df['status'].str.contains('In Progress')
    ].copy()
    
    if len(in_progress) == 0:
        st.warning("No in-progress activities found")
        return
    
    st.info(f"Analyzing {len(in_progress)} in-progress activities...")
    
    # Make predictions
    predictions = predictor.predict(in_progress)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    critical = (predictions['risk_level'] == 'Critical Risk').sum()
    high = (predictions['risk_level'] == 'High Risk').sum()
    medium = (predictions['risk_level'] == 'Medium Risk').sum()
    low = (predictions['risk_level'] == 'Low Risk').sum()
    
    with col1:
        st.markdown('<div class="risk-critical">🚨 Critical Risk</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="big-metric">{critical}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="risk-high">⚠️ High Risk</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="big-metric">{high}</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="risk-medium">⚡ Medium Risk</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="big-metric">{medium}</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="risk-low">✅ Low Risk</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="big-metric">{low}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        risk_filter = st.multiselect(
            "Filter by Risk Level",
            options=['Critical Risk', 'High Risk', 'Medium Risk', 'Low Risk'],
            default=['Critical Risk', 'High Risk']
        )
    with col2:
        system_filter = st.multiselect(
            "Filter by System Type",
            options=predictions['system_type'].unique(),
            default=predictions['system_type'].unique()
        )
    
    # Apply filters
    filtered = predictions[
        (predictions['risk_level'].isin(risk_filter)) &
        (predictions['system_type'].isin(system_filter))
    ].sort_values('delay_probability', ascending=False)
    
    # Display high-risk activities
    st.subheader(f"🚨 High-Risk Activities ({len(filtered)} activities)")
    
    display_cols = [
        'activity_id', 'equipment_type', 'system_type', 'complexity',
        'contractor', 'delay_probability', 'risk_level'
    ]
    
    display_df = filtered[display_cols].copy()
    display_df['delay_probability'] = (display_df['delay_probability'] * 100).round(1).astype(str) + '%'
    
    # Color code risk levels
    def highlight_risk(row):
        if row['risk_level'] == 'Critical Risk':
            return ['background-color: #ff4444; color: white'] * len(row)
        elif row['risk_level'] == 'High Risk':
            return ['background-color: #ff9800; color: white'] * len(row)
        elif row['risk_level'] == 'Medium Risk':
            return ['background-color: #ffeb3b; color: black'] * len(row)
        else:
            return ['background-color: #4caf50; color: white'] * len(row)
    
    st.dataframe(
        display_df.style.apply(highlight_risk, axis=1),
        use_container_width=True,
        height=400
    )
    
    # Export button
    csv = filtered.to_csv(index=False)
    st.download_button(
        label="📥 Export Predictions to CSV",
        data=csv,
        file_name=f"commissioniq_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

def show_analytics(activities_df, predictor):
    """Show detailed analytics"""
    
    st.header("📈 Advanced Analytics")
    
    # Time-based analysis
    st.subheader("📅 Delay Trends Over Time")
    
    activities_df['planned_start_date'] = pd.to_datetime(activities_df['planned_start_date'])
    activities_df['month'] = activities_df['planned_start_date'].dt.to_period('M').astype(str)
    
    monthly_delays = activities_df.groupby('month').agg({
        'is_delayed': ['sum', 'count', 'mean']
    }).reset_index()
    monthly_delays.columns = ['Month', 'Delayed', 'Total', 'Delay Rate']
    monthly_delays['Delay Rate'] = monthly_delays['Delay Rate'] * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_delays['Month'],
        y=monthly_delays['Delay Rate'],
        mode='lines+markers',
        name='Delay Rate',
        line=dict(color='red', width=3)
    ))
    fig.update_layout(
        title='Monthly Delay Rate Trend',
        xaxis_title='Month',
        yaxis_title='Delay Rate (%)',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Contractor performance
    st.markdown("---")
    st.subheader("👷 Contractor Performance Analysis")
    
    contractor_perf = activities_df.groupby('contractor').agg({
        'is_delayed': ['sum', 'count', 'mean'],
        'delay_days': 'mean',
        'contractor_performance_score': 'mean'
    }).reset_index()
    contractor_perf.columns = ['Contractor', 'Delayed', 'Total', 'Delay Rate', 'Avg Delay Days', 'Performance Score']
    contractor_perf['Delay Rate'] = (contractor_perf['Delay Rate'] * 100).round(1)
    contractor_perf = contractor_perf.sort_values('Delay Rate', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            contractor_perf,
            x='Contractor',
            y='Delay Rate',
            color='Delay Rate',
            color_continuous_scale='RdYlGn_r',
            labels={'Delay Rate': 'Delay Rate (%)'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            contractor_perf,
            x='Performance Score',
            y='Delay Rate',
            size='Total',
            color='Avg Delay Days',
            hover_data=['Contractor'],
            color_continuous_scale='Reds',
            labels={
                'Performance Score': 'Historical Performance Score',
                'Delay Rate': 'Delay Rate (%)',
                'Total': 'Number of Activities'
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def show_single_predictor(predictor):
    """Interactive single activity predictor"""
    
    st.header("⚙️ Single Activity Risk Assessment")
    st.markdown("Enter activity details to get instant delay prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        equipment_type = st.selectbox(
            "Equipment Type",
            options=['HVAC Chiller', 'Air Handler', 'Boiler', 'Cooling Tower', 'Pump', 
                    'Fan', 'Electrical Switchgear', 'Generator', 'UPS', 'Transformer']
        )
        
        system_type = st.selectbox(
            "System Type",
            options=['HVAC', 'Electrical', 'Plumbing', 'Fire Protection', 'BMS']
        )
        
        activity_type = st.selectbox(
            "Activity Type",
            options=['Equipment Delivery', 'Installation', 'Startup', 
                    'Functional Testing', 'Performance Testing', 'Documentation Review']
        )
        
        complexity = st.selectbox(
            "Complexity",
            options=['Low', 'Medium', 'High', 'Critical']
        )
        
        contractor = st.selectbox(
            "Contractor",
            options=['Johnson Controls', 'Siemens', 'Trane', 'Carrier', 
                    'ABB', 'Schneider Electric', 'Honeywell', 'Eaton']
        )
    
    with col2:
        planned_duration = st.number_input(
            "Planned Duration (days)",
            min_value=1,
            max_value=30,
            value=7
        )
        
        num_dependencies = st.slider(
            "Number of Dependencies",
            min_value=0,
            max_value=10,
            value=2
        )
        
        contractor_score = st.slider(
            "Contractor Performance Score",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.05
        )
        
        doc_completeness = st.slider(
            "Document Completeness (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.85,
            step=0.05
        )
        
        resource_availability = st.selectbox(
            "Resource Availability",
            options=['Full', 'Partial', 'Limited']
        )
        
        weather_sensitive = st.checkbox("Weather Sensitive")
        prior_experience = st.checkbox("Prior Similar Experience", value=True)
    
    if st.button("🔮 Predict Delay Risk", type="primary"):
        # Create activity data
        complexity_score = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}[complexity]
        
        activity_data = {
            'equipment_type': equipment_type,
            'system_type': system_type,
            'activity_type': activity_type,
            'complexity': complexity,
            'complexity_score': complexity_score,
            'planned_duration_days': planned_duration,
            'num_dependencies': num_dependencies,
            'contractor': contractor,
            'contractor_performance_score': contractor_score,
            'resource_availability': resource_availability,
            'document_completeness_pct': doc_completeness,
            'weather_sensitive': weather_sensitive,
            'prior_similar_experience': prior_experience
        }
        
        # Make prediction
        result = predictor.predict(pd.DataFrame([activity_data]))
        
        delay_prob = result['delay_probability'].values[0]
        risk_level = result['risk_level'].values[0]
        
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        
        # Display result
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Delay Probability", f"{delay_prob*100:.1f}%")
        with col2:
            risk_color = {
                'Low Risk': '🟢',
                'Medium Risk': '🟡',
                'High Risk': '🟠',
                'Critical Risk': '🔴'
            }
            st.metric("Risk Level", f"{risk_color[risk_level]} {risk_level}")
        with col3:
            recommendation = "Immediate Action Required" if delay_prob > 0.7 else \
                           "Monitor Closely" if delay_prob > 0.5 else \
                           "Standard Monitoring"
            st.metric("Recommendation", recommendation)
        
        # Risk factors
        st.markdown("---")
        st.subheader("⚠️ Key Risk Factors")
        
        risk_factors = []
        if contractor_score < 0.75:
            risk_factors.append("Low contractor performance score")
        if doc_completeness < 0.7:
            risk_factors.append("Incomplete documentation")
        if num_dependencies > 3:
            risk_factors.append("High number of dependencies")
        if resource_availability == 'Limited':
            risk_factors.append("Limited resource availability")
        if complexity_score >= 3:
            risk_factors.append("High complexity activity")
        if not prior_experience:
            risk_factors.append("No prior similar experience")
        
        if risk_factors:
            for factor in risk_factors:
                st.warning(f"⚠️ {factor}")
        else:
            st.success("✅ No major risk factors identified")

if __name__ == '__main__':
    main()
