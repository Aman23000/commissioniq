#!/usr/bin/env python3
"""
CommissionIQ Dashboard Launcher
Run this script from the project root directory
"""

import sys
import os

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Change to project directory
os.chdir(current_dir)

# Import streamlit and run
import streamlit.web.cli as stcli

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", "app/dashboard.py"]
    sys.exit(stcli.main())
