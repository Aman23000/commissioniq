"""
Generate Realistic Commissioning Dataset
Simulates data that Facility Grid would collect from construction projects
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
NUM_PROJECTS = 50
NUM_ACTIVITIES_PER_PROJECT = 200

# Reference data
PROJECT_TYPES = ['Data Center', 'Hospital', 'Airport', 'Office Building', 'Manufacturing', 'Higher Education']
EQUIPMENT_TYPES = ['HVAC Chiller', 'Air Handler', 'Boiler', 'Cooling Tower', 'Pump', 'Fan', 'Electrical Switchgear', 
                   'Generator', 'UPS', 'Transformer', 'Fire Alarm Panel', 'BMS Controller', 'VAV Box', 'Exhaust Fan']
SYSTEM_TYPES = ['HVAC', 'Electrical', 'Plumbing', 'Fire Protection', 'BMS']
ACTIVITY_TYPES = ['Equipment Delivery', 'Installation', 'Startup', 'Functional Testing', 'Performance Testing', 'Documentation Review']
CONTRACTORS = ['Johnson Controls', 'Siemens', 'Trane', 'Carrier', 'ABB', 'Schneider Electric', 'Honeywell', 'Eaton']
LOCATIONS = ['Boston, MA', 'New York, NY', 'San Francisco, CA', 'Austin, TX', 'Seattle, WA', 'Chicago, IL', 'Dallas, TX']

def generate_projects():
    """Generate project metadata"""
    projects = []
    
    for i in range(NUM_PROJECTS):
        project = {
            'project_id': f'PRJ-{i+1:03d}',
            'project_name': f'{random.choice(PROJECT_TYPES)} - {random.choice(LOCATIONS).split(",")[0]} Phase {random.randint(1,3)}',
            'project_type': random.choice(PROJECT_TYPES),
            'location': random.choice(LOCATIONS),
            'project_value_millions': round(random.uniform(10, 500), 1),
            'planned_duration_days': random.randint(180, 730),
            'start_date': datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365)),
            'contractor': random.choice(CONTRACTORS),
            'building_size_sqft': random.randint(50000, 2000000)
        }
        projects.append(project)
    
    return pd.DataFrame(projects)

def generate_commissioning_activities(projects_df):
    """Generate commissioning activities for each project"""
    activities = []
    
    for _, project in projects_df.iterrows():
        project_start = project['start_date']
        num_activities = random.randint(150, 250)
        
        for j in range(num_activities):
            equipment_type = random.choice(EQUIPMENT_TYPES)
            system = random.choice(SYSTEM_TYPES)
            activity_type = random.choice(ACTIVITY_TYPES)
            
            # Planned dates
            planned_start = project_start + timedelta(days=random.randint(30, project['planned_duration_days']-60))
            planned_duration = random.randint(1, 14)
            planned_end = planned_start + timedelta(days=planned_duration)
            
            # Complexity scoring (affects delay probability)
            complexity = random.choice(['Low', 'Medium', 'High', 'Critical'])
            complexity_score = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}[complexity]
            
            # Dependencies
            num_dependencies = random.randint(0, 5) if j > 10 else 0
            
            # Weather sensitivity (for HVAC outdoor equipment)
            weather_sensitive = equipment_type in ['Cooling Tower', 'Generator', 'Chiller'] and random.random() > 0.5
            
            # Contractor performance score (historical)
            contractor_score = round(random.uniform(0.6, 1.0), 2)
            
            # Resource availability
            resource_availability = random.choice(['Full', 'Partial', 'Limited'])
            
            # Document completeness at activity start
            doc_completeness = round(random.uniform(0.4, 1.0), 2)
            
            # Prior similar work experience
            prior_experience = random.choice([True, False])
            
            # Actual execution (determine if delayed)
            # Delay probability factors
            delay_prob = 0.3  # Base probability
            
            # Increase probability based on risk factors
            if complexity_score >= 3:
                delay_prob += 0.15
            if num_dependencies > 3:
                delay_prob += 0.1
            if contractor_score < 0.75:
                delay_prob += 0.15
            if resource_availability == 'Limited':
                delay_prob += 0.2
            elif resource_availability == 'Partial':
                delay_prob += 0.1
            if doc_completeness < 0.7:
                delay_prob += 0.15
            if not prior_experience:
                delay_prob += 0.1
            if weather_sensitive:
                delay_prob += 0.05
                
            # Cap at 0.95
            delay_prob = min(delay_prob, 0.95)
            
            is_delayed = random.random() < delay_prob
            
            if is_delayed:
                delay_days = random.randint(1, 21)
                actual_start = planned_start + timedelta(days=random.randint(-2, 5))
                actual_end = planned_end + timedelta(days=delay_days)
                status = 'Completed - Delayed' if random.random() > 0.2 else 'In Progress - At Risk'
            else:
                actual_start = planned_start + timedelta(days=random.randint(-1, 2))
                actual_end = planned_end + timedelta(days=random.randint(-2, 2))
                delay_days = 0
                status = 'Completed - On Time' if random.random() > 0.3 else 'In Progress - On Track'
            
            # Issues encountered
            num_issues = random.randint(0, 8) if is_delayed else random.randint(0, 2)
            
            # Change orders
            change_orders = random.randint(0, 3) if is_delayed else random.randint(0, 1)
            
            activity = {
                'activity_id': f'{project["project_id"]}-ACT-{j+1:04d}',
                'project_id': project['project_id'],
                'equipment_id': f'{equipment_type.replace(" ", "-")}-{j+1:03d}',
                'equipment_type': equipment_type,
                'system_type': system,
                'activity_type': activity_type,
                'complexity': complexity,
                'complexity_score': complexity_score,
                'planned_start_date': planned_start,
                'planned_end_date': planned_end,
                'planned_duration_days': planned_duration,
                'actual_start_date': actual_start if status.startswith('Completed') else None,
                'actual_end_date': actual_end if status == 'Completed - On Time' or status == 'Completed - Delayed' else None,
                'status': status,
                'is_delayed': is_delayed,
                'delay_days': delay_days,
                'num_dependencies': num_dependencies,
                'contractor': project['contractor'],
                'contractor_performance_score': contractor_score,
                'resource_availability': resource_availability,
                'document_completeness_pct': doc_completeness,
                'weather_sensitive': weather_sensitive,
                'prior_similar_experience': prior_experience,
                'num_issues_encountered': num_issues,
                'num_change_orders': change_orders,
                'budget_variance_pct': round(random.uniform(-0.1, 0.3) if is_delayed else random.uniform(-0.05, 0.1), 3)
            }
            
            activities.append(activity)
    
    return pd.DataFrame(activities)

def main():
    """Generate and save datasets"""
    print("Generating commissioning dataset...")
    
    # Generate data
    projects_df = generate_projects()
    activities_df = generate_commissioning_activities(projects_df)
    
    # Save to CSV
    projects_df.to_csv('/home/claude/commissioniq/data/projects.csv', index=False)
    activities_df.to_csv('/home/claude/commissioniq/data/commissioning_activities.csv', index=False)
    
    # Print summary statistics
    print(f"\n✅ Dataset Generated Successfully!")
    print(f"\nProjects: {len(projects_df)}")
    print(f"Total Activities: {len(activities_df)}")
    print(f"\nDelay Statistics:")
    print(f"  - Delayed Activities: {activities_df['is_delayed'].sum()} ({activities_df['is_delayed'].mean()*100:.1f}%)")
    print(f"  - On-Time Activities: {(~activities_df['is_delayed']).sum()} ({(~activities_df['is_delayed']).mean()*100:.1f}%)")
    print(f"  - Average Delay (when delayed): {activities_df[activities_df['is_delayed']]['delay_days'].mean():.1f} days")
    
    print(f"\nProject Type Distribution:")
    print(projects_df['project_type'].value_counts())
    
    print(f"\nActivity Status Distribution:")
    print(activities_df['status'].value_counts())
    
    print(f"\nComplexity Distribution:")
    print(activities_df['complexity'].value_counts())

if __name__ == '__main__':
    main()
