# Time-Series Analysis of Political Violence Targeting Women (PVTW)

This project performs a time-series analysis on the **Political Violence Targeting Women (PVTW)** dataset, derived from the **ACLED** database. The goal is to aggregate and analyze events of political violence and fatalities targeting women across different countries over time.

## Project Structure

```
TimeSeriesPVTW/
├── utils/                         # All suppor code for the project
│   ├── result_reviewer.py         # Scripts for data preprocessing and transformations│   
│   ├── time_series_pvtw.py        # Main module containing TimeSeriesPVTW class to process the data
│
├── data/                          # Data used in the project
│   ├── raw/                       # Original, unprocessed data
│   │   └── ACLED_DATA.csv         # Raw data (ACLED dataset) containing political violence events
│   ├── processed/                 # Cleaned and preprocessed data
│       └── ts_pvtw.csv            # Processed time-series data (generated after analysis)
│       └── README.md              # A description of the dataset(s) and how to use them
│
├── references/                    # State-of-the-art reference papers related to time series transformers
│
├── results/                       # Results of the project (e.g., output, figures, logs)
│   ├── figures/                   # Plots and visualizations
│   ├── logs/                      # Logs for training or evaluation (e.g., model outputs)
│   └── final_model/               # Final trained model (e.g., .pth, .h5, or other formats)
│
├── notebooks/                     # Jupyter notebooks (if used for exploratory analysis or experiments)
│   └── analysis.ipynb             # A sample Jupyter notebook for analysis and experimentation
│
├── run_exp.py                     # Main script to run the project
├── README.md                      # Project overview, setup instructions, and guidelines
├── requirements.txt               # Python dependencies and environment setup
└── LICENSE                        # Project license

```

## Project Overview

This project is designed to process, analyzie and predict political violence events targeting women, as well as related fatalities, from the **ACLED** dataset. The data is aggregated into a time-series format, summarizing fatalities and event counts per country on each day.

### Key Features:
- **Time-Series Analysis**: Aggregates the number of fatalities and counts of events on a daily basis for each country.
- **Data Processing**: Processes the raw ACLED dataset into a time-series format, focusing on the variables related to political violence targeting women.
- **Command-Line Interface (CLI)**: The analysis can be run using a Python script from the command line, allowing flexible input/output paths.

## Installation

To set up and run this project, follow the steps below:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/TimeSeriesPVTW.git
   cd TimeSeriesPVTW
