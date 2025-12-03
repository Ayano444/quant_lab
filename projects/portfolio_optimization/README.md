# Portfolio Optimization Engine (Markowitz Modern Portfolio Theory)

This project implements a complete, real-world portfolio optimization engine using Python.  
It covers the full Markowitz framework, including:

- Market data loading (Yahoo Finance)
- Log return computation
- Portfolio statistics (return, variance, volatility, Sharpe ratio)
- Global Minimum Variance (GMV) Portfolio
- Maximum Sharpe Ratio (MSR) Portfolio
- Efficient Frontier construction
- Visualization of portfolios and risk–return tradeoff



---

## 📌 1. Project Goal

Build a clean, reproducible portfolio optimization framework that:
- fetches market data,
- computes returns,
- performs convex optimization,
- generates efficient frontier plots,
- summarizes portfolio performance.

This is the foundation of quantitative asset management.

---

## 📈 2. Methods & Concepts

### **Modern Portfolio Theory (MPT)**
A mathematical framework for constructing portfolios that maximize expected return for a given level of risk.

### **Global Minimum Variance (GMV) Portfolio**
The portfolio with the lowest possible volatility.

### **Maximum Sharpe Ratio (MSR) Portfolio**
The portfolio that maximizes:
\[
\text{Sharpe Ratio} = \frac{R_p - R_f}{\sigma_p}
\]

### **Efficient Frontier**
The set of optimal portfolios with the best risk–return tradeoff.

---

## 🧮 3. Project Structure

portfolio_optimization/
│── data/
│── notebook.ipynb
│── README.md
└── sample_plots/

The optimization functions are stored in:
utils/quant_functions.py

---

## 🧰 4. Technologies Used

- Python
- NumPy
- Pandas
- yfinance
- CVXPY (convex optimization)
- Matplotlib
- Jupyter Notebook

---

## 🔍 5. How to Run

1. Install dependencies:
pip install numpy pandas yfinance cvxpy matplotlib

2. Open the notebook:

projects/portfolio_optimization/notebook.ipynb

3. Run all cells.  
It will:
- download price data  
- compute returns  
- compute GMV & MSR portfolios  
- generate the efficient frontier  

---

## 📊 6. Sample Output

You can include saved PNG outputs here:
sample_plots/efficient_frontier.png

---

## 📝 7. Results Summary

The notebook prints:
- GMV weights + expected return + volatility + Sharpe  
- MSR weights + expected return + volatility + Sharpe  
- Efficient frontier plot  

---

## 🚀 8. Future Enhancements

- Streamlit UI for interactive portfolio selection
- Support for more assets
- Allow short-selling options
- Integrate factor models (Fama-French)
- Include Black-Litterman allocation

---

## ✉️ Author
**Mannu Raj Shrivastava**  
Quant & Data Science Student  
GitHub: https://github.com/Ayano444

