# US Immigration Analysis Web App

This package was generated from the uploaded notebook and CSV.

## Files
- `app.py` - Streamlit web app entry point
- `utils.py` - data processing, modeling, and plotting helpers
- `requirements.txt` - Python dependencies
- `state_data_2013-2023_20250514_3.csv` - dataset
- `041726_Tianle_Immigration_Analysis.ipynb` - original notebook

## How to run

### 1) Unzip
Extract the zip to a folder.

### 2) Install dependencies
Open a terminal in that folder and run:

```bash
pip install -r requirements.txt
```

### 3) Launch the app
```bash
streamlit run app.py
```

### 4) Open browser
If it does not open automatically, go to:
`http://localhost:8501`

## Notes
- The `Dense NN` and `LSTM` tabs require TensorFlow and may take longer to run.
- The app uses a time-based split: train years `<= 2021`, test years `>= 2022`.
- The app is designed to be a practical conversion of the notebook, not a pixel-perfect notebook clone.
