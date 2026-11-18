# students-risk-prediction


# Initial file structure

/project-2-risk-prediction
├── data/
│   ├── raw/                 # Original OULAD CSVs (git-ignored)
│   └── processed/           # Merged datasets ready for training
├── src/
│   ├── data_prep/
│   │   ├── feature_engineering.py
│   │   └── temporal_cutoffs.py  # Generates the 20/40/60/80/100% data splits
│   ├── models/
│   │   ├── train_models.py      # Script to run Random Forest, XGBoost, SVM, LR, NB
│   │   └── evaluate.py          # 25x 10-fold CV logic, SMOTE, and Metric calculation
│   └── visualize/
│       └── plot_metrics.py      # ROC-AUC graphs and ablation study plots
├── notebooks/
│   └── eda.ipynb            # Jupyter notebook for initial data exploration
├── requirements.txt         # List of libraries
└── README.md                # Project setup instructions
