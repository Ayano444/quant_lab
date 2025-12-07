QuantLab – Quantitative Finance Research & Web Applications

QuantLab is a collection of quantitative finance tools, research notebooks, and a full-featured web application for portfolio optimization.
Built using Python, Django, NumPy, Pandas, SciPy, Matplotlib, and modern web UI components.

This repository represents a complete ecosystem of quant research + deployable analytics.

🚀 Features
### 1. Portfolio Optimization Web App (Django)

Located in:

quantweb/


A full-stack portfolio optimization dashboard with:

Classic Modern Portfolio Theory

Global Minimum Variance (GMV)

Maximum Sharpe Ratio (MSR)

Efficient Frontier Plotting

Portfolio Weights Table

Annualized Returns, Volatility, Sharpe Ratio

Correlation Matrix (auto-computed)

High-quality UI with gradients & animations

Risk Parity Optimizer (coming soon)

Hierarchical Risk Parity (HRP) (coming soon)

All results are dynamically generated from Yahoo Finance price data.

📊 2. Quantitative Research & Experiments

Located in:

notebooks/


Contains Jupyter notebooks for:

Portfolio optimization theory

Monte Carlo simulation engines

Random returns modeling

Asset allocation experiments

Risk analysis modules

Option pricing prototypes

Time-series forecasting demos

This folder shows your mathematical and research capability.

🧰 3. Core Utilities

Located in:

utils/


Reusable tools used across the project:

Covariance calculations

Optimization wrappers

Sharpe / volatility helpers

Data preprocessing utilities

Risk model helpers

These utilities power both the research notebooks and the Django web app.

🏗 Tech Stack
Backend

Python 3.11+

Django 4+

NumPy

Pandas

SciPy

Matplotlib / Seaborn

Frontend

HTML5 / CSS3

Modern UI gradients & card-based layouts

Interactive dashboards

Data

Yahoo Finance API for live & historical pricing

📦 Installation

Clone the repo:

git clone https://github.com/Ayano444/quant_lab.git
cd quant_lab


Create and activate environment:

conda create -n quant_lab python=3.11
conda activate quant_lab
pip install -r requirements.txt


Run Django app:

cd quantweb
python manage.py runserver


Visit in browser:

http://127.0.0.1:8000/

🧠 Why This Project Matters

This repository demonstrates:

Strong Python + Django engineering skills

Real quantitative finance knowledge

Ability to build full-stack analytics systems

Practical exposure to portfolio theory

Research + application in one combined repo

It is a portfolio-ready project for quant internships, finance roles, and tech + data science interviews.

👤 Author

Mannu
GitHub: Ayano444

Built with curiosity, math, and caffeine.

📜 License

MIT License – free for personal and educational use.
