NSE SmartTrade AI - lightweight Windows-friendly prototype

Setup (PowerShell, from this folder):
  py -m pip install -r requirements.txt
  py -m streamlit run app.py

This version intentionally avoids scikit-learn/SciPy because a Windows Application Control policy on some systems blocks their compiled DLLs. Its prediction button uses a simple least-squares trend baseline, not Random Forest. It is educational and not investment advice.
