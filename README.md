# Introduction

The repository contains the code implementation of the paper:

Exploring Large Language Models for Indoor Occupancy Detection and Estimation for Smart Buildings

<p align="center"> <img src="Figures/CoT_FewShot.jpg" alt="LLM Occupancy Framework" width="700"/> </p>

This study proposes an LLM-based occupancy detection and estimation framework using few-shot learning, chain-of-thought, and in-context learning, demonstrating that models like Gemini-Pro and DeepSeek-R1 outperform traditional methods across diverse datasets from China and Singapore, offering robust and adaptable solutions for smart building management.

If you find this project helpful, please give us a star ⭐ — your support is our greatest motivation.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Model Training](#model-training)
- [Model Testing](#model-testing)
- [Image Prediction](#image-prediction)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)
- [Contact Us](#contact-us)


## Installation
### Dependencies
- Windows / Linux / macOS  
- Python 3.8+  
- Required Python packages: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `requests`  
- [Ollama](https://ollama.com/) (for local LLaMA 3.2 model inference)  
- API access for **DeepSeek-R1** and **Gemini-Pro**

### Environment Installation
We recommend using **Miniconda** to manage your Python environment.
**Step 0:** Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
**Step 1:** Create and activate a virtual environment
```bash
conda create -n llm_occ python=3.8 -y
conda activate llm_occ
```
**Step 3:** Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib requests
```

### LLM Setup
Install [Ollama 3.2](https://ollama.com/) and pull the LLaMA model:

```bash
ollama pull llama3:8b
```
Our llm_models.py script connects to the local Ollama instance for LLaMA-based predictions.

Run the model locally:

bash
Copy
Edit
ollama run llama3
Our llm_models.py script connects to the local Ollama instance for LLaMA-based predictions.

🔹 DeepSeek-R1 & Gemini-Pro (via API Keys)
Obtain your API keys from the following sources:

DeepSeek API

Gemini API

Set your keys in a .env file or your environment:

bash
Copy
Edit
# .env file format
DEEPSEEK_API_KEY=your_deepseek_key
GEMINI_API_KEY=your_gemini_key
Clone the Repository
bash
Copy
Edit
git clone https://github.com/kailaisun/LLM-occucpancy.git
cd LLM-occucpancy




- **Data Processing**: Resamples time-series data into 5, 10, and 30-minute intervals.
- **Normalization**: Standardizes features using `StandardScaler`.
- **Data Splitting**: Splits data into weekly ranges and filters for office hours (9 AM–6 PM).
- **Data Balancing**: Undersamples the majority class to balance occupancy data.
- **Models**: Includes LLMs (llama3.2:latest, Gemini-1.5-pro, DeepSeek_R1) and baseline models (Logistic Regression, Random Forest, Decision Tree, XGBoost).

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/LLM-occupancy.git
   cd LLM-occupancy

2. ## Create a Conda Environment:

 ```bash
conda create --name ml_env python=3.8
conda activate ml_env
conda install pandas scikit-learn numpy requests
 ```

3. ## Prepare Your Data:
Place your CSV files in a folder named Dataset within the project directory.
Ensure the CSV files have a datetime column in "YYYY-MM-DD HH:MM:SS" format and a binary occupant_num column.

# Usage
To run the project, follow these steps:

1. ## Run Data Processing and Normalization:
- Update input_file_path and resampled_paths in the scripts to match your file names.
- Execute:
 ```bash
python data_processing.py
python normalization.py
 ```
2. ## Data Splitting:
- Splits data into weekly ranges and filters for office hours (9 AM–6 PM)
- Run:
 ```bash
python Data_Splitting.py
 ```
   
4. ## Balance the Dataset:
- Modify input_file_path in the balancing script.
- Run:
 ```bash
python data_balancing.py
 ```
4. ## Run LLM Models:
- Ensure the LLM API is hosted locally (e.g., Ollama for llama3.2:latest).
- Adjust the model parameter in generate_llm_response() for other LLMs.
- Execute:
 ```bash
python llm_models.py
 ```
5. ## Run Baseline Models:
- Ensure processed CSV files are in the Dataset folder.
- Run:
 ```bash
python baseline_models.py
 ```
# Model Accuracy Results

| Strategy | Week  | Interval | LR     | RF     | DT     | XGBoost | LLaMA 3.2 | Gemini-Pro | DeepSeek-R1 | RI (%) |
|-----------|-------|-----------|--------|--------|--------|---------|------------|-------------|--------------|--------|
| 1 | Week 1 | 5 min  | **89.04%** | 80.73% | 61.46% | 55.15% | 87.04% | **89.04%** | 86.93% | **0.00%** |
|   |        | 10 min | 85.45% | 79.39% | 51.30% | 64.85% | 86.06% | **93.33%** | 83.64% | **9.22%** |
|   |        | 30 min | 72.97% | 86.49% | 64.86% | 81.08% | 86.49% | **88.89%** | 79.73% | **2.77%** |
|  | Week 2 | 5 min   | 90.75% | 84.83% | 71.21% | 69.41% | 89.72% | **92.03%** | 81.75% | **1.41%** |
|   |        | 10 min | 89.29% | 89.29% | 65.48% | 82.14% | 85.12% | **91.67%** | 90.48% | **2.67%** |
|   |        | 30 min | 77.14% | 85.71% | 77.14% | 77.14% | 88.57% | **91.43%** | 88.57% | **6.67%** |
|  | Week 3 | 5 min   | 52.65% | 60.68% | 54.06% | 63.60% | 84.81% | **93.64%** | 92.23% | **49.57%** |
|   |        | 10 min | 53.50% | 55.41% | 52.23% | 54.78% | 85.99% | **94.27%** | 92.90% | **70.13%** |
|   |        | 30 min | 50.00% | 55.26% | 50.00% | 55.26% | 89.47% | **90.79%** | 88.89% | **64.30%** |
| 2 | Week 1 | 5 min  | 91.19% | 90.75% | 91.63% | 91.63% | 81.94% | **95.59%** | **96.04%** | **4.81%** |
|   |        | 10 min | 78.91% | 88.28% | 79.69% | 85.16% | 85.94% | **94.53%** | 92.97% | **7.08%** |
|   |        | 30 min | 78.18% | 81.82% | 80.00% | 80.00% | 87.27% | **90.91%** | 88.68% | **11.11%** |
|  | Week 2 | 5 min   | **92.73%** | 89.27% | 74.05% | 83.74% | 84.08% | 92.04% | 91.70% | **-0.74%** |
|   |        | 10 min | 87.70% | 91.80% | 85.25% | 90.98% | 86.07% | **94.26%** | 93.44% | **2.68%** |
|   |        | 30 min | 79.17% | 87.50% | **91.67%** | 75.00% | 87.50% | **91.67%** | 83.33% | **0.00%** |
|  | Week 3 | 5 min   | 50.93% | 57.87% | 63.43% | 59.26% | 85.19% | **92.59%** | 86.06% | **45.97%** |
|   |        | 10 min | 50.86% | 56.90% | 51.72% | 52.59% | 86.21% | **93.10%** | 89.47% | **63.62%** |
|   |        | 30 min | 50.88% | 57.89% | 59.65% | 59.65% | 89.47% | **92.98%** | 85.71% | **55.88%** |
| 3 | Week 1 | 5 min  | 90.65% | 90.65% | 84.89% | 82.01% | 84.17% | **94.96%** | 90.37% | **4.75%** |
|   |        | 10 min | 85.00% | 85.00% | 81.25% | 82.50% | 86.25% | **95.00%** | 89.33% | **11.76%** |
|   |        | 30 min | 88.89% | 83.33% | 72.22% | 80.56% | 88.89% | **94.44%** | 91.67% | **6.24%** |
|  | Week 2 | 5 min   | 90.26% | 92.82% | 71.28% | 90.77% | 82.05% | 95.90% | **96.41%** | **3.87%** |
|   |        | 10 min | 90.80% | 90.80% | 85.06% | **91.95%** | 85.06% | 90.80% | 91.67% | **-0.30%** |
|   |        | 30 min | 80.95% | 85.71% | 71.43% | 90.48% | 85.71% | 90.48% | **95.24%** | **5.26%** |
|  | Week 3 | 5 min   | 63.82% | 63.82% | 58.55% | 61.84% | 82.89% | 90.79% | **91.61%** | **43.54%** |
|   |        | 10 min | 54.65% | 62.79% | 58.14% | 65.12% | 82.56% | **93.02%** | 90.70% | **42.84%** |
|   |        | 30 min | 63.16% | 60.53% | 52.63% | 57.89% | 84.21% | **94.74%** | 92.11% | **50.00%** |
| 4 | Week 1 | 5 min  | 91.67% | **95.83%** | 87.50% | 94.44% | 91.67% | **95.83%** | 94.29% | **0.00%** |
|   |        | 10 min | 86.36% | **90.91%** | 68.18% | 88.64% | 86.36% | 88.64% | 86.05% | **-2.50%** |
|   |        | 30 min | 83.33% | 83.33% | 44.44% | 83.33% | **88.89%** | **88.89%** | 83.33% | **6.67%** |
|  | Week 2 | 5 min   | 86.73% | 91.84% | 74.49% | 86.73% | 86.73% | 94.90% | **95.92%** | **4.44%** |
|   |        | 10 min | 87.50% | 87.50% | 75.00% | **91.67%** | 89.58% | 87.50% | 86.96% | **-2.28%** |
|   |        | 30 min | 91.67% | **94.50%** | 66.67% | **94.50%** | 91.67% | 91.67% | 90.91% | **-2.99%** |
|  | Week 3 | 5 min   | 76.71% | 65.75% | 57.53% | 63.01% | 82.19% | **89.04%** | 86.57% | **16.07%** |
|   |        | 10 min | 65.12% | 60.47% | 53.49% | 62.79% | 86.05% | **93.02%** | 90.00% | **42.84%** |
|   |        | 30 min | 63.16% | 63.16% | 63.16% | 63.16% | **94.74%** | 89.47% | 84.21% | **50.00%** |



**Note**: Results shown are for llama3.2:latest. To generate results for Gemini-1.5-pro or DeepSeek_R1, adjust the model parameter in the LLM script.




# Contributing
Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch:
```bash
git checkout -b feature/new-feature
```
3. Commit your changes:
```bash
git commit -m 'Add new feature'
```
4. Push to the branch:
```bash
git push origin feature/new-feature
```
5. Open a pull request.
Please ensure your code adheres to the project’s style and includes tests where applicable.


## Citation

This paper is under review. The citation format is coming soon.


## Contact Us

If you have other questions❓, please contact us in time 👬
