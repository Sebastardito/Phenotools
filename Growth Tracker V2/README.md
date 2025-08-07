# LarvaeTracker - Larvae Growth Analysis Platform

## Overview
LarvaeTracker is a comprehensive web-based application designed for analyzing and visualizing larvae growth data. Built with Python (Flask backend) and modern web technologies (HTML/CSS/JS frontend), it provides researchers with powerful tools for statistical analysis, trend detection, and comparative studies of larvae development metrics.

## Key Features
- **Interactive Growth Visualization**: Plot growth curves with customizable standard deviations
- **Multi-level Statistical Analysis**: Calculate statistics by population, cycle, group, and combinations
- **Correlation Analysis**: Compute Pearson and Spearman correlations between different groups
- **Trend Detection**: Identify growth trends and classify their strength
- **Comparative Analysis**: Compare maximum/minimum values across selected groups
- **Statistical Testing**: Perform ANOVA and Tukey HSD post-hoc tests
- **Cluster Analysis**: Identify similar growth patterns using hierarchical clustering
- **Automated Reporting**: Generate comprehensive PDF reports with visualizations and statistics
- **Responsive Dashboard**: Modern UI with dark/light theme support

## Technical Requirements
- Python 3.7+
- Required Python packages:
  - Flask
  - pandas
  - matplotlib
  - scipy
  - seaborn
  - statsmodels
  - reportlab

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/larvae-tracker.git
cd larvae-tracker
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Run the application with your CSV data file:
```bash
python tracker.py path/to/your/data.csv
```

2. The application will automatically open your default browser to:
```
http://127.0.0.1:5000
```

3. In the web interface:
   - Select populations, cycles, and groups to analyze
   - Choose a growth metric (e.g., Indv_Weight)
   - Adjust visualization settings (standard deviations, annotations)
   - Explore various statistical analyses through the dashboard tabs
   - Generate PDF reports with the "Download PDF Report" button

## Input Data Format
Your CSV should contain these required columns:
- `Population`: Population identifier
- `Cycle`: Growth cycle identifier
- `Group`: Experimental group identifier
- `Age_Days`: Age in days
- `Data_Date`: Date of measurement (optional)
- One or more measurement columns (e.g., `Indv_Weight`)

Example CSV structure:
```
Population,Cycle,Group,Age_Days,Data_Date,Indv_Weight,Other_Metric
PopA,Cycle1,Group1,10,2024-03-01,0.25,15.2
PopA,Cycle1,Group1,15,2024-03-06,0.42,18.7
PopB,Cycle2,Group3,10,2024-03-01,0.28,16.1
...
```

## Application Structure
```
larvae-tracker/
├── tracker.py            # Main Flask application
├── templates/
│   └── index.html        # Dashboard HTML template
├── requirements.txt      # Python dependencies
└── data.csv              # Example data file (not included)
```

## Statistical Methods Implemented
- Descriptive statistics (mean, std dev, percentiles)
- Linear regression for trend analysis
- Pearson and Spearman correlations
- ANOVA with post-hoc Tukey HSD tests
- Hierarchical clustering
- Daily growth rate calculations
- Comparative analysis with percentage differences
- Hypothesis testing (ANOVA)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author
Sebastardito! - SAEU 2025
