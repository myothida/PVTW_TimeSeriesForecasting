# Time-Series Analysis of Political Violence Targeting Women (PVTW)

This project performs a time-series analysis on the **Political Violence Targeting Women (PVTW)** dataset, derived from the **ACLED** database. The goal is to aggregate and analyze events of political violence and fatalities targeting women across different countries over time.

## Project Structure

```
PVTW_TimeSeriesForecasting/
├── utils/                         # Supporting code for the project
│   ├── plot_helper.py             # Functions to visualize data and results
│   ├── sequentialdataset.py       # Handles sequential dataset creation for training
│   ├── time_series_pvtw.py        # Functions specific to analyze the PVTW dataset 
│   ├── time_series_stl.py         # STL decomposition methods for time series
│
├── data/                          # Dataset-related files
│   ├── raw/                       # Original, unprocessed data
│   │   ├── ACLED_DATA.csv         # Raw ACLED dataset
│   │   ├── ACLED_data.md          # Documentation for the ACLED dataset
│   │   └── acled_metadata.csv     # Metadata file for the ACLED dataset
│   ├── processed/                 # Cleaned and preprocessed data
│
├── layers/                        # Core components for transformer-based models
│   ├── attention.py               # Implements attention mechanisms
│   ├── decomposition.py           # Methods for decomposition-based forecasting
│   ├── embedding.py               # Embedding layer implementation
│   ├── transformer_encoder.py     # Transformer encoder implementation
│   └── transformer_decoder.py     # Transformer decoder implementation
│
├── models/                        # Machine learning model implementations
│   ├── nf_linears.py              # Neural forecasting with linear models
│   ├── nf_mlp.py                  # Neural forecasting with multi-layer perceptrons
│   ├── nf_transformer.py          # Neural forecasting with transformers
│   └── timeseries_transformer.py  # Implementation of the proposed time series transformer
│
├── results/                       # Project outputs and results
│   ├── figures/                   # Visualizations and plots
│   ├── data/                      # Logs for training and evaluation
│   └── final_model/               # Saved models (e.g., .pth files)
│
├── scripts/                       # Standalone scripts for running models
│   └── pvtw_tsf_transformer.py    # Main script for training and evaluating the transformer model
│
├── notebooks/                     # Jupyter notebooks for analysis and experimentation
│   ├── run_analysis.ipynb         # Exploratory data analysis (EDA)
│   ├── run_comparison.ipynb       # Model comparison and benchmarking
│   ├── run_nf_models.ipynb        # Neural forecasting model training and evaluation
│   └── run_TSTransform.ipynb      # Time series transformer experiments
│
├── README.md                      # Project overview and usage instructions
├── requirements.txt               # Python dependencies for the project
├── LICENSE                        # Project license
└── run_exp.py                     # Placeholder for running experiments or models


```

### Key Features:
- **Time-Series Analysis**: Aggregates the number of fatalities and counts of events on a daily basis for each country.
- **Data Processing**: Processes the raw ACLED dataset into a time-series format, focusing on the variables related to political violence targeting women.

### Installation

To set up and run this project, follow the steps below:

1. Clone the repository:
   ```bash
   git clone https://github.com/myothida/01_TimeSeriesForConflict.git
   cd 01_TimeSeriesForConflict
   run run_TSTransform.ipynb

### Acknowledgments
This research utilizes the ACLED dataset, downloaded on December 23, 2024. The project aims to advance the understanding of gender-targeted political violence using machine learning methods.

#### Declaration and Privacy
The data used in this project is sourced from the **Armed Conflict Location and Event Data Project (ACLED)**. All users of this dataset must adhere to ACLED’s **Terms of Use** and **Privacy Policy**. Redistribution, modification, or commercial use of this dataset without proper permission from ACLED is strictly prohibited.

For more information about ACLED's data usage policies, please visit their official website: [ACLED Data Terms and Conditions](https://acleddata.com).