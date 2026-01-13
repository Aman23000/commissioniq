#!/usr/bin/env python3
"""
CommissionIQ - Simple Command Line Demo
Use this if Streamlit has issues
"""

import sys
import os

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'models'))

import pandas as pd
from train_model import CommissionDelayPredictor

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(text.center(70))
    print("=" * 70 + "\n")

def main():
    print_header("COMMISSIONIQ - PREDICTIVE DELAY ANALYSIS")
    
    # Load model
    print("📊 Loading trained model...")
    predictor = CommissionDelayPredictor()
    model_path = os.path.join(current_dir, 'models/')
    predictor.load_model(model_path)
    print("✅ Model loaded successfully!\n")
    
    # Load data
    print("📁 Loading commissioning data...")
    data_path = os.path.join(current_dir, 'data', 'commissioning_activities.csv')
    activities_df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(activities_df)} commissioning activities\n")
    
    # Get in-progress activities
    in_progress = activities_df[activities_df['status'].str.contains('In Progress')].copy()
    print(f"🔍 Found {len(in_progress)} in-progress activities\n")
    
    # Make predictions
    print("🤖 Running AI predictions...")
    predictions = predictor.predict(in_progress)
    
    # Summary statistics
    print_header("RISK ASSESSMENT SUMMARY")
    
    risk_counts = predictions['risk_level'].value_counts()
    
    print(f"🔴 CRITICAL RISK: {risk_counts.get('Critical Risk', 0)} activities")
    print(f"🟠 HIGH RISK:     {risk_counts.get('High Risk', 0)} activities")
    print(f"🟡 MEDIUM RISK:   {risk_counts.get('Medium Risk', 0)} activities")
    print(f"🟢 LOW RISK:      {risk_counts.get('Low Risk', 0)} activities")
    
    # Show top 10 highest risk activities
    print_header("TOP 10 HIGHEST RISK ACTIVITIES")
    
    top_risks = predictions.nlargest(10, 'delay_probability')
    
    for idx, row in top_risks.iterrows():
        print(f"\n{'─' * 70}")
        print(f"Activity ID:      {row['activity_id']}")
        print(f"Equipment:        {row['equipment_type']}")
        print(f"System:           {row['system_type']}")
        print(f"Complexity:       {row['complexity']}")
        print(f"Contractor:       {row['contractor']}")
        print(f"📊 Delay Risk:    {row['delay_probability']*100:.1f}%")
        print(f"⚠️  Risk Level:    {row['risk_level']}")
    
    # Overall metrics
    print_header("PORTFOLIO METRICS")
    
    delayed = activities_df['is_delayed'].sum()
    total = len(activities_df)
    delay_rate = (delayed / total) * 100
    avg_delay = activities_df[activities_df['is_delayed']]['delay_days'].mean()
    
    print(f"Total Activities:        {total:,}")
    print(f"Delayed Activities:      {delayed:,}")
    print(f"Overall Delay Rate:      {delay_rate:.1f}%")
    print(f"Average Delay Duration:  {avg_delay:.1f} days")
    
    # Feature importance
    print_header("TOP 5 RISK FACTORS (AI Model)")
    
    for idx, row in predictor.feature_importance.head(5).iterrows():
        feature_name = row['feature'].replace('_encoded', '').replace('_', ' ').title()
        importance = row['importance'] * 100
        print(f"{idx+1}. {feature_name:<30} {importance:>5.1f}%")
    
    # Cost impact estimate
    print_header("ESTIMATED COST IMPACT")
    
    avg_cost_per_delay = 55000  # $5K-10K per day × 11 days average
    potential_savings = len(predictions[predictions['risk_level'].isin(['Critical Risk', 'High Risk'])]) * avg_cost_per_delay * 0.7  # 70% accuracy
    
    print(f"High-Risk Activities:     {len(predictions[predictions['risk_level'].isin(['Critical Risk', 'High Risk'])])}")
    print(f"Avg Cost per Delay:       ${avg_cost_per_delay:,}")
    print(f"AI Detection Accuracy:    70%")
    print(f"💰 Potential Savings:     ${potential_savings:,.0f}")
    print(f"\n(Based on preventing delays through early intervention)")
    
    # Export option
    print_header("EXPORT DATA")
    
    export = input("\n📥 Export high-risk activities to CSV? (y/n): ").lower()
    
    if export == 'y':
        high_risk = predictions[predictions['risk_level'].isin(['Critical Risk', 'High Risk'])]
        output_file = 'high_risk_activities.csv'
        high_risk.to_csv(output_file, index=False)
        print(f"✅ Exported {len(high_risk)} activities to '{output_file}'")
    
    print_header("DEMO COMPLETE")
    print("This demonstrates CommissionIQ's predictive capabilities.")
    print("For the full interactive dashboard, run: streamlit run app/dashboard.py")
    print("\n")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you're running from the project root directory:")
        print("  cd commissioniq")
        print("  python demo.py")
        sys.exit(1)
