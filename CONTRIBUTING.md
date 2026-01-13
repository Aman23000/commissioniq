# Contributing to CommissionIQ

Thank you for your interest in contributing to CommissionIQ!

## 🎯 Project Purpose

This project was created as a demonstration of AI-powered predictive analytics for construction commissioning. While it's primarily a portfolio/demonstration project, contributions that improve the model, add features, or enhance documentation are welcome.

## 🤝 How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. Check if the issue already exists
2. Open a new issue with:
   - Clear description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Your environment (OS, Python version, etc.)

### Submitting Changes

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Test thoroughly**
5. **Commit with clear messages**
   ```bash
   git commit -m "Add feature: description"
   ```
6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Open a Pull Request**

## 🎨 Areas for Contribution

### Model Improvements
- [ ] Try different ML algorithms (XGBoost, LightGBM, Neural Networks)
- [ ] Add SHAP for better explainability
- [ ] Implement ensemble methods
- [ ] Add cross-validation
- [ ] Hyperparameter tuning

### Features
- [ ] Real-time prediction API
- [ ] Email/Slack alert integrations
- [ ] PDF report generation
- [ ] Integration with Procore/BIM 360
- [ ] Mobile app interface
- [ ] Historical trend analysis

### Data
- [ ] Add more realistic data generation patterns
- [ ] Create benchmark datasets
- [ ] Add data validation tools

### Documentation
- [ ] Video tutorials
- [ ] Use case examples
- [ ] API documentation
- [ ] Deployment guides

### Testing
- [ ] Unit tests for model components
- [ ] Integration tests
- [ ] Performance benchmarks

## 📝 Code Style

- Follow PEP 8 guidelines
- Add docstrings to functions
- Comment complex logic
- Keep functions focused and small
- Use type hints where helpful

## 🧪 Testing

Before submitting:
```bash
# Test model training
python models/train_model.py

# Test dashboard
python run_dashboard.py

# Test CLI demo
python demo.py
```

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be acknowledged in the README and release notes.

## 💬 Questions?

Feel free to open an issue for any questions about contributing!

---

**Thank you for helping improve CommissionIQ! 🚀**
