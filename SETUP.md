# ⚡ Quick Setup Guide

## Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

## Installation (5 minutes)

### 1. Unzip the Project
Extract the `commissioniq.zip` file to your preferred location.

### 2. Navigate to the Project Folder
```bash
cd commissioniq
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

If you get permission errors, try:
```bash
pip install --user -r requirements.txt
```

### 4. Verify Installation - Test the Model
```bash
python models/train_model.py
```

This will train the model and display performance metrics. You should see:
- ✅ Accuracy: ~70%
- ✅ ROC-AUC: ~0.685

### 5. Run Dashboard (Option A - Recommended)
```bash
python run_dashboard.py
```

### 5. Run Dashboard (Option B - Manual)
```bash
streamlit run app/dashboard.py
```

### 5. Run Dashboard (Option C - Simple CLI Demo)
If Streamlit has issues, use the command-line demo:
```bash
python demo.py
```

The dashboard will open in your browser at `http://localhost:8501` (Options A & B)  
The CLI demo runs in your terminal (Option C)

---

## Project Structure

```
commissioniq/
├── README.md                          # Full documentation
├── START_HERE.md                      # Your action plan
├── SETUP.md                           # This file
├── run_dashboard.py                   # Easy launcher
├── requirements.txt                   # Python dependencies
│
├── data/
│   ├── generate_data.py              # Synthetic data generator
│   ├── projects.csv                   # 50 construction projects
│   └── commissioning_activities.csv   # 10,207 activities
│
├── models/
│   ├── train_model.py                 # ML training pipeline
│   ├── delay_predictor.pkl            # Trained Random Forest model
│   ├── label_encoders.pkl             # Feature encoders
│   ├── feature_names.pkl              # Feature list
│   └── feature_importance.csv         # Feature importance rankings
│
└── app/
    └── dashboard.py                   # Streamlit web interface
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'numpy._core'"
**Solution:** This is a numpy/streamlit compatibility issue. Try:
```bash
pip install numpy==1.26.4 streamlit==1.28.0
```

Or use the CLI demo instead:
```bash
python demo.py
```

### Issue: "ModuleNotFoundError: No module named 'streamlit'"
**Solution:** Install the missing packages:
```bash
pip install streamlit pandas numpy scikit-learn plotly
```

### Issue: "ModuleNotFoundError: No module named 'train_model'"
**Solution:** Make sure you're running from the project root directory:
```bash
cd commissioniq
python run_dashboard.py
```

Or use the full path:
```bash
cd commissioniq
streamlit run app/dashboard.py
```

### Issue: Dashboard shows import errors
**Solution:** The paths are now relative, so you must run from the project root:
```bash
# CORRECT (from commissioniq folder)
cd commissioniq
python run_dashboard.py

# WRONG (from inside app folder)
cd commissioniq/app
streamlit run dashboard.py  # This won't work!
```

### Issue: Model accuracy seems different
**Solution:** The data is randomly generated, so exact numbers may vary slightly (65-75% is normal).

### Issue: Permission denied when installing packages
**Solution:** Install for your user only:
```bash
pip install --user -r requirements.txt
```

---

## Quick Commands Reference

```bash
# Navigate to project
cd commissioniq

# Install everything
pip install -r requirements.txt

# Test the model
python models/train_model.py

# Launch dashboard (easy way)
python run_dashboard.py

# Launch dashboard (manual way)
streamlit run app/dashboard.py

# Regenerate data (optional)
python data/generate_data.py
```

---

## Demo Checklist

Before presenting to Facility Grid:

- [ ] Run `python models/train_model.py` to verify model works
- [ ] Launch dashboard with `python run_dashboard.py`
- [ ] Test all 4 tabs (Overview, Predictions, Analytics, Predictor)
- [ ] Practice 15-minute walkthrough
- [ ] Have backup screenshots ready
- [ ] Read DEMO_GUIDE.md thoroughly
- [ ] Prepare answers to common questions

---

## What Each File Does

**Documentation:**
- `START_HERE.md` - Your complete action plan (read this first!)
- `README.md` - Full technical and business documentation
- `DEMO_GUIDE.md` - Presentation script with Q&A
- `EXECUTIVE_SUMMARY.md` - Quick overview
- `ONE_PAGE_PITCH.md` - Business case summary

**Code:**
- `run_dashboard.py` - Easy launcher script (NEW!)
- `app/dashboard.py` - Web interface
- `models/train_model.py` - ML model training
- `data/generate_data.py` - Data generator

**Data:**
- `data/projects.csv` - Project information
- `data/commissioning_activities.csv` - Activity records
- `models/*.pkl` - Trained model files

---

## Next Steps

1. ✅ Install and test locally
2. ✅ Read START_HERE.md for your action plan
3. ✅ Practice the demo
4. ✅ Contact your friend at Facility Grid TODAY!

---

## Need Help?

If something doesn't work:
1. Make sure you're in the `commissioniq` folder
2. Try `python run_dashboard.py` instead of the manual command
3. Check that all dependencies are installed
4. Verify Python version is 3.8+

---

**Good luck! You've got this! 🚀**

---

## Project Structure

```
commissioniq/
├── README.md                          # Full documentation
├── DEMO_GUIDE.md                      # Presentation script
├── PROJECT_SUMMARY.md                 # Next steps guide
├── requirements.txt                   # Python dependencies
│
├── data/
│   ├── generate_data.py              # Synthetic data generator
│   ├── projects.csv                   # 50 construction projects
│   └── commissioning_activities.csv   # 10,207 activities
│
├── models/
│   ├── train_model.py                 # ML training pipeline
│   ├── delay_predictor.pkl            # Trained Random Forest model
│   ├── label_encoders.pkl             # Feature encoders
│   ├── feature_names.pkl              # Feature list
│   └── feature_importance.csv         # Feature importance rankings
│
└── app/
    └── dashboard.py                   # Streamlit web interface
```

---

## Troubleshooting

### Issue: "Module not found" errors
**Solution:** Install missing packages individually:
```bash
pip install streamlit pandas numpy scikit-learn plotly
```

### Issue: Dashboard won't load
**Solution:** Make sure you're in the project root directory:
```bash
cd commissioniq
streamlit run app/dashboard.py
```

### Issue: Model accuracy seems different
**Solution:** The data is randomly generated, so exact numbers may vary slightly. As long as accuracy is 65-75%, everything is working correctly.

---

## Demo Checklist

Before presenting to Facility Grid:

- [ ] Run `python models/train_model.py` to verify model works
- [ ] Launch dashboard and test all 4 tabs
- [ ] Practice 15-minute walkthrough
- [ ] Have backup screenshots ready
- [ ] Read DEMO_GUIDE.md thoroughly
- [ ] Prepare answers to common questions

---

## Files to Share

When sharing this project with Facility Grid:

**Essential:**
- ✅ README.md (comprehensive documentation)
- ✅ All code files (.py)
- ✅ Trained model files (.pkl, .csv)
- ✅ Sample data (projects.csv, activities.csv)

**Optional:**
- DEMO_GUIDE.md (if they want presentation tips)
- PROJECT_SUMMARY.md (if they want setup instructions)

**How to share:**
1. **GitHub** (recommended): Upload to private repo, share link
2. **ZIP file**: Compress entire folder, email/share via Google Drive
3. **Live demo**: Run locally and screen share

---

## Next Steps

1. ✅ Test everything works locally
2. ✅ Practice demo walkthrough
3. ✅ Contact your friend at Facility Grid
4. ✅ Schedule presentation
5. ✅ Nail the interview!

---

## Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Read the full README.md for detailed explanations
3. Google error messages (most are common Python issues)
4. Reach out to me if needed

---

**Good luck! You've got a killer project here. 🚀**
