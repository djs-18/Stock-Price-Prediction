# 📈 Stock Price Prediction System using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)

A machine learning-powered web application built with Streamlit to predict stock prices using Support Vector Regression (SVR). Users can upload stock data, visualize trends, and get real-time price predictions.

---

## 🎯 Project Overview

This project provides an interactive platform for stock price prediction featuring:

- 📊 **Interactive Data Upload** - CSV file support for stock data
- 📈 **Real-time Visualization** - Dynamic charts using Plotly
- 🤖 **ML Model Training** - SVR with linear kernel
- 🎯 **Price Prediction** - Real-time stock price forecasting
- 📋 **Model Evaluation** - R², MAE, MSE, RMSE metrics

---

## 🚀 Features

- **Data Processing**: Automatic handling of missing values and date formatting
- **Feature Selection**: Choose input features for model training
- **Model Training**: SVR algorithm with standardized features
- **Performance Metrics**: Comprehensive model evaluation
- **Interactive Predictions**: Real-time price prediction with user inputs
- **Visualization**: Actual vs Predicted price comparisons

---

## 🛠️ Technologies Used

- **Python** - Core programming language
- **Streamlit** - Web application framework
- **Scikit-learn** - Machine learning library
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Plotly** - Interactive visualizations
- **Matplotlib** - Static plotting

---

## 📁 Project Structure

```
Stock-Price-Prediction/
├── streamlit_app.py                 # Main Streamlit application
├── Stock_Price_Prediction.ipynb     # Jupyter notebook with analysis
├── ITC.NS.csv                      # Sample stock data (ITC stock)
├── Stock_Price_Prediction.pdf      # Project documentation
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/djs-18/Stock-Price-Prediction.git
cd Stock-Price-Prediction

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate         # On Windows
# source venv/bin/activate     # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Application

```bash
# Start the Streamlit app
streamlit run streamlit_app.py
```

The application will open at **http://localhost:8501**

---

## 📊 How to Use

1. **Upload Data**: Use the sidebar to upload your CSV stock data file
2. **Data Preview**: View the uploaded dataset and basic statistics
3. **Visualization**: Explore stock price trends over time
4. **Feature Selection**: Choose input features for model training
5. **Model Training**: Automatic SVR model training with data preprocessing
6. **Evaluation**: View model performance metrics (R², MAE, MSE, RMSE)
7. **Prediction**: Input values in the sidebar to get real-time predictions

---

## 📈 Model Performance

The SVR model provides:
- **Algorithm**: Support Vector Regression with linear kernel
- **Features**: Open, High, Low, Volume prices
- **Target**: Close price prediction
- **Preprocessing**: StandardScaler for feature normalization
- **Evaluation**: Multiple metrics for comprehensive assessment

---

## 📋 Sample Data Format

Your CSV file should contain columns:
```
Date, Open, High, Low, Close, Volume, Adj Close
```

Sample data (ITC.NS.csv) is included for testing.

---

## 🔮 Future Enhancements

- [ ] Multiple ML algorithms comparison
- [ ] Real-time data fetching from APIs
- [ ] Technical indicators integration
- [ ] Portfolio optimization features
- [ ] Advanced time series forecasting
- [ ] Model deployment on cloud platforms

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

This application is for educational and research purposes only. Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always consult with financial professionals before making investment choices.

---

## 👨‍💻 Author

**Your Name**
- GitHub: [@djs-18](https://github.com/djs-18)
- LinkedIn: [Your LinkedIn Profile]

---

**📅 Last Updated**: August 2025 | **🔢 Version**: 1.0.0