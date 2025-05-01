# AUTOMATED-EMAIL-SUBJECT-LINE-GENERATION-USING-DEEP-LEARNING
A Sequence-to-Sequence Approach for Abstractive Summarization 

**Author:** Anitha R  | **Date:** May 2, 2025
*IIT Madras, ADSML Cohort 6 - Group 12*

---

## 1. Introduction

Crafting effective email subjects is crucial for communication but often time-consuming. This project explores the use of deep learning, specifically sequence-to-sequence Transformer models, to automatically generate concise and relevant subject lines from the content of email bodies.

The project involved:
*   Processing the Annotated Enron Subject Line Corpus (AESLC).
*   Implementing a robust data cleaning and preprocessing pipeline.
*   Experimenting with T5-small and fine-tuning BART-large models using the Hugging Face ecosystem (`transformers`, `datasets`, `evaluate`) on Google Colab.
*   Evaluating model performance using automated ROUGE metrics and structured human evaluation (Relevance, Conciseness, Fluency).
*   Developing an interactive demonstration application using Streamlit.

## 2. Features

*   End-to-end pipeline for subject line generation from raw data to model training and evaluation.
*   Comparative analysis of T5-small vs. BART-large performance.
*   Implementation of both automated (ROUGE) and human evaluation methodologies.
*   Refined data cleaning process addressing PII, boilerplate, and specific noise.
*   Functional Streamlit demo app showcasing the best performing model (`facebook/bart-large`).

## 3. Dataset

*   **Name:** Annotated Enron Subject Line Corpus (AESLC)
*   **Source:** Derived from the public Enron Email Corpus.
*   **Link:** [https://github.com/ryanzhumich/AESLC](https://github.com/ryanzhumich/AESLC)
*   **Description:** Contains real-world email bodies and corresponding subject lines. The training set (~14.4k emails) has original subjects, while the development (~1.9k) and test (~1.9k) sets include multiple human annotations, enabling robust evaluation. The dataset is known for its highly abstractive nature (low subject-body overlap).

## 4. Methodology Overview

The project followed these core steps:

1.  **Initial Data Loading:** Raw AESLC text files were parsed and structured into CSV format (Script 1).
2.  **Data Cleaning & EDA:** Extensive cleaning (PII removal, boilerplate reduction, noise removal, normalization) was applied, and basic EDA was performed (Script 2). Original and cleaned data were saved.
3.  **Model Preprocessing:** Cleaned data was tokenized specifically for the target model (`facebook/bart-large`) using its `AutoTokenizer`, handling truncation and formatting for sequence-to-sequence input (Script 3). Processed data was saved in Arrow format.
4.  **Model Training:** The pre-trained `facebook/bart-large` model was fine-tuned on the processed training data using the Hugging Face `Seq2SeqTrainer` on Google Colab, validating against the development set using ROUGE metrics (Script 4 / Notebook). The best checkpoint was saved. (An initial experiment with `t5-small` was also conducted for comparison).
5.  **Prediction & Human Eval Prep:** Subject lines were generated for the test set using the best model. A sample was prepared for human evaluation, comparing model output against original subjects and human annotations (Script 5).
6.  **Human Evaluation Analysis:** Scores from human raters (Relevance, Conciseness, Fluency) were collected and analyzed, including calculating average scores and Inter-Annotator Agreement (Script 6).
7.  **Demonstration:** A Streamlit application was built to provide an interactive interface for the fine-tuned model (Script 7 - `app.py`).

## 5. Repository Structure


├── scripts/ # Directory containing the processing/training scripts
│ ├── 1_...Data_Load...py
│ ├── 2_...Clean_EDA...py
│ ├── 3_...Preprocessing...py
│ ├── 4_...Model_Training...py # Or link to Colab Notebook
│ ├── 5...Generate_Human_Eval_Prep...py
│ ├── 6...Analyze_Human_Eval...py
│ └── app...py # Streamlit App
├── notebooks/ # Optional: Jupyter notebooks for EDA or experimentation
├── data/ # Output directory for intermediate CSVs (cleaned), processed data, logs, etc. (gitignore recommended)
│ ├── CLEANED_DATA/ # Output from Script 2
│ ├── processed_datasets/ # Output from Script 3 (Arrow format)
│ └── training_output/ # Output from Script 4 (Models, checkpoints, logs)
│ └── Human_Evaluation/ # Output from Scripts 5 & 6 (Eval sheets, analysis)
├── raw_data/ # Recommended: Place downloaded AESLC raw data here (gitignore recommended)
│ └── enron_subject_line/
│ ├── train/
│ ├── dev/
│ └── test/
├── .gitignore # To exclude data, virtual environments, etc.
├── README.md # This file
├── requirements.txt # Python dependencies
└── report/ # Optional: Location for the final project report PDF

## 6. Setup Instructions

**Prerequisites:**

*   Python (>= 3.9 recommended)
*   `pip` package installer
*   Git (for cloning)
*   Access to a machine with a GUI for running scripts 1, 2, 3, 5, 6, 7 (which use Tkinter prompts).
*   Access to Google Colab with GPU for running Step 4 (Model Training).

**Steps:**

1.  **Clone Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate (Linux/macOS)
    source venv/bin/activate
    # Activate (Windows)
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You may need to create `requirements.txt` first using `pip freeze > requirements.txt` in your final working environment)*. Ensure `torch` is installed correctly for your system (CPU or specific CUDA version).

4.  **Download NLTK Data:** Run this command once in your activated environment:
    ```bash
    python -m nltk.downloader punkt punkt_tab stopwords
    ```

5.  **Download AESLC Dataset:**
    *   Clone or download the AESLC dataset from [https://github.com/ryanzhumich/AESLC](https://github.com/ryanzhumich/AESLC).
    *   Place the extracted `enron_subject_line` directory (containing `train`, `dev`, `test` subfolders with `.subject` files) inside the `raw_data/` directory within your cloned project folder. Your structure should look like `YourProject/raw_data/enron_subject_line/train/...`.

## 7. Usage Instructions

Run the scripts sequentially. Most scripts (1, 2, 3, 5, 6, and the Streamlit app) will prompt you graphically (using Tkinter) to select input files/folders and output locations.

1.  **Script 1: Initial Data Load & Structuring**
    *   **Purpose:** Parses raw `.subject` files into initial CSVs.
    *   **Run:** `python scripts/1_..._Data_Load_...py`
    *   **Input:** Prompts for raw `train`, `dev`, `test` directories (select folders inside `raw_data/enron_subject_line/`). Prompts for an output directory (e.g., create `data/INITIAL_CSV/`).
    *   **Output:** Timestamped CSV files (e.g., `train_sample_14436_....csv`) in the chosen output directory.

2.  **Script 2: Cleaning & EDA**
    *   **Purpose:** Applies refined cleaning rules to the CSVs from Step 1.
    *   **Run:** `python scripts/2_..._Clean_EDA_...py`
    *   **Input:** Prompts to select the `train`, `dev`, `test` CSV files generated in Step 1. Prompts for an output directory (e.g., `data/CLEANED_DATA/`).
    *   **Output:** Timestamped CSV files (`*_cleaned_*.csv`) containing original and cleaned columns, plus logs.

3.  **Script 3: Model Preprocessing**
    *   **Purpose:** Tokenizes cleaned data for the chosen model (`facebook/bart-large`).
    *   **Run:** `python scripts/3_..._Preprocessing_...py`
    *   **Input:** Prompts to select the `train`, `dev`, `test` cleaned CSV files from Step 2. Prompts for a base output directory (e.g., `data/`).
    *   **Output:** Saves tokenized data in Arrow format to a subdirectory named `processed_datasets` within the chosen base output directory. Also saves logs.

4.  **Script 4: Model Training (Run on Colab with GPU)**
    *   **Purpose:** Fine-tunes the `facebook/bart-large` model.
    *   **Action:**
        *   Upload the `processed_datasets` directory (from Step 3) to your Google Drive.
        *   Open the corresponding script/notebook (`4_..._Model_Training...`) in Google Colab.
        *   Ensure the runtime type is set to GPU.
        *   Modify the path constants inside the notebook (Cell 4) to point to your `processed_datasets` directory on Drive and define where model outputs should be saved on Drive.
        *   Run the notebook cells sequentially. This step takes significant time (~1-2 hours).
    *   **Output:** Fine-tuned model checkpoints, logs, and metric files saved to the specified Google Drive output directory.

5.  **Script 5: Generate Predictions & Prep Human Eval**
    *   **Purpose:** Uses the best trained model to generate test set predictions and creates rating sheets.
    *   **Action:** *(Run locally after downloading the best model checkpoint from Drive from Step 4)*.
    *   **Run:** `python scripts/5_..._Generate_Human_Eval_Prep...py`
    *   **Input:** Prompts to select the best model checkpoint directory (downloaded from Drive). Prompts to select the cleaned test CSV (from Step 2). Prompts for an output directory (e.g., `data/Human_Evaluation/`).
    *   **Output:** Prediction file (`model_test_predictions_*.csv`), rater input sheet (`human_evaluation_sheet_input_*.csv`), keyed sheet (`human_evaluation_sheet_keyed_*.csv`), logs.

6.  **Step 6: Conduct Human Evaluation (Manual)**
    *   Distribute the `human_evaluation_sheet_input_*.csv` to 2-3 raters with clear instructions.
    *   Collect the completed rating sheets (e.g., save as `rater1_eval.csv`, `rater2_eval.csv`).

7.  **Script 7: Analyze Human Evaluation Results**
    *   **Purpose:** Calculates average scores and IAA from rater sheets.
    *   **Run:** `python scripts/6_..._Analyze_Human_Eval...py`
    *   **Input:** Prompts for the directory containing completed rater CSVs (from Step 6). Prompts for the keyed sheet CSV (from Step 5). Prompts for an output directory.
    *   **Output:** Summary statistics CSV, IAA scores CSV, qualitative examples CSV, logs.

## 8. Results Summary

*   **Model Comparison:** BART-large significantly outperformed T5-small on ROUGE metrics.
*   **Final BART-large Test Performance:** Achieved **ROUGE-L: 0.3460** on the test set.
*   **Human Evaluation:** Showed moderate rater agreement (Kappa ~0.4-0.5). Model-generated subjects were rated as generally good (Avg scores > 3.1/4), but lower than human-written subjects (Avg scores ~3.5-3.8), especially on conciseness.
*   **Generated Length:** The best model produced a median subject length of 22 tokens on the test set, longer than the human average (~4).

*For detailed results and analysis, please refer to the full project report.*

## 9. Demo (Streamlit App)

An interactive demo application is provided in `scripts/app_...py`.

**To Run the Demo Locally:**

1.  **Update Model Path:** Edit the `MODEL_PATH` variable inside the Streamlit script (`scripts/app_...py`) to point to the directory containing your downloaded best fine-tuned BART-large model checkpoint (from Step 4).
2.  **Run:** Execute the following command in your terminal (ensure you are in the project's root directory and your virtual environment is active):
    ```bash
    streamlit run scripts/app_...py
    ```
3.  Open the provided local URL (e.g., `http://localhost:8501`) in your web browser.



[Screenshot or GIF of Streamlit App - see output folder]

## 10. License

All Rights Reserved . See LICENSE file for more

## 11. Acknowledgements

*   Dataset: Annotated Enron Subject Line Corpus (AESLC) by Zhang and Tetreault.
*   Libraries: Hugging Face (Transformers, Datasets, Evaluate), PyTorch, Pandas, NLTK, Streamlit, Scikit-learn.
*   Compute: Google Colab.



├── scripts/ # Directory containing the processing/training scripts
│ ├── 1_...Data_Load...py
│ ├── 2_...Clean_EDA...py
│ ├── 3_...Preprocessing...py
│ ├── 4_...Model_Training...py # Or link to Colab Notebook
│ ├── 5...Generate_Human_Eval_Prep...py
│ ├── 6...Analyze_Human_Eval...py
│ └── app...py # Streamlit App
├── notebooks/ # Optional: Jupyter notebooks for EDA or experimentation
├── data/ # Output directory for intermediate CSVs (cleaned), processed data, logs, etc. (gitignore recommended)
│ ├── CLEANED_DATA/ # Output from Script 2
│ ├── processed_datasets/ # Output from Script 3 (Arrow format)
│ └── training_output/ # Output from Script 4 (Models, checkpoints, logs)
│ └── Human_Evaluation/ # Output from Scripts 5 & 6 (Eval sheets, analysis)
├── raw_data/ # Recommended: Place downloaded AESLC raw data here (gitignore recommended)
│ └── enron_subject_line/
│ ├── train/
│ ├── dev/
│ └── test/
├── .gitignore # To exclude data, virtual environments, etc.
├── README.md # This file
├── requirements.txt # Python dependencies
└── report/ # Optional: Location for the final project report PDF


## 6. Setup Instructions

**Prerequisites:**

*   Python (>= 3.9 recommended)
*   `pip` package installer
*   Git (for cloning)
*   Access to a machine with a GUI for running scripts 1, 2, 3, 5, 6, 7 (which use Tkinter prompts).
*   Access to Google Colab with GPU for running Step 4 (Model Training).

**Steps:**

1.  **Clone Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate (Linux/macOS)
    source venv/bin/activate
    # Activate (Windows)
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You may need to create `requirements.txt` first using `pip freeze > requirements.txt` in your final working environment)*. Ensure `torch` is installed correctly for your system (CPU or specific CUDA version).

4.  **Download NLTK Data:** Run this command once in your activated environment:
    ```bash
    python -m nltk.downloader punkt punkt_tab stopwords
    ```

5.  **Download AESLC Dataset:**
    *   Clone or download the AESLC dataset from [https://github.com/ryanzhumich/AESLC](https://github.com/ryanzhumich/AESLC).
    *   Place the extracted `enron_subject_line` directory (containing `train`, `dev`, `test` subfolders with `.subject` files) inside the `raw_data/` directory within your cloned project folder. Your structure should look like `YourProject/raw_data/enron_subject_line/train/...`.

## 7. Usage Instructions

Run the scripts sequentially. Most scripts (1, 2, 3, 5, 6, and the Streamlit app) will prompt you graphically (using Tkinter) to select input files/folders and output locations.

1.  **Script 1: Initial Data Load & Structuring**
    *   **Purpose:** Parses raw `.subject` files into initial CSVs.
    *   **Run:** `python scripts/1_..._Data_Load_...py`
    *   **Input:** Prompts for raw `train`, `dev`, `test` directories (select folders inside `raw_data/enron_subject_line/`). Prompts for an output directory (e.g., create `data/INITIAL_CSV/`).
    *   **Output:** Timestamped CSV files (e.g., `train_sample_14436_....csv`) in the chosen output directory.

2.  **Script 2: Cleaning & EDA**
    *   **Purpose:** Applies refined cleaning rules to the CSVs from Step 1.
    *   **Run:** `python scripts/2_..._Clean_EDA_...py`
    *   **Input:** Prompts to select the `train`, `dev`, `test` CSV files generated in Step 1. Prompts for an output directory (e.g., `data/CLEANED_DATA/`).
    *   **Output:** Timestamped CSV files (`*_cleaned_*.csv`) containing original and cleaned columns, plus logs.

3.  **Script 3: Model Preprocessing**
    *   **Purpose:** Tokenizes cleaned data for the chosen model (`facebook/bart-large`).
    *   **Run:** `python scripts/3_..._Preprocessing_...py`
    *   **Input:** Prompts to select the `train`, `dev`, `test` cleaned CSV files from Step 2. Prompts for a base output directory (e.g., `data/`).
    *   **Output:** Saves tokenized data in Arrow format to a subdirectory named `processed_datasets` within the chosen base output directory. Also saves logs.

4.  **Script 4: Model Training (Run on Colab with GPU)**
    *   **Purpose:** Fine-tunes the `facebook/bart-large` model.
    *   **Action:**
        *   Upload the `processed_datasets` directory (from Step 3) to your Google Drive.
        *   Open the corresponding script/notebook (`4_..._Model_Training...`) in Google Colab.
        *   Ensure the runtime type is set to GPU.
        *   Modify the path constants inside the notebook (Cell 4) to point to your `processed_datasets` directory on Drive and define where model outputs should be saved on Drive.
        *   Run the notebook cells sequentially. This step takes significant time (~1-2 hours).
    *   **Output:** Fine-tuned model checkpoints, logs, and metric files saved to the specified Google Drive output directory.

5.  **Script 5: Generate Predictions & Prep Human Eval**
    *   **Purpose:** Uses the best trained model to generate test set predictions and creates rating sheets.
    *   **Action:** *(Run locally after downloading the best model checkpoint from Drive from Step 4)*.
    *   **Run:** `python scripts/5_..._Generate_Human_Eval_Prep...py`
    *   **Input:** Prompts to select the best model checkpoint directory (downloaded from Drive). Prompts to select the cleaned test CSV (from Step 2). Prompts for an output directory (e.g., `data/Human_Evaluation/`).
    *   **Output:** Prediction file (`model_test_predictions_*.csv`), rater input sheet (`human_evaluation_sheet_input_*.csv`), keyed sheet (`human_evaluation_sheet_keyed_*.csv`), logs.

6.  **Step 6: Conduct Human Evaluation (Manual)**
    *   Distribute the `human_evaluation_sheet_input_*.csv` to 2-3 raters with clear instructions.
    *   Collect the completed rating sheets (e.g., save as `rater1_eval.csv`, `rater2_eval.csv`).

7.  **Script 7: Analyze Human Evaluation Results**
    *   **Purpose:** Calculates average scores and IAA from rater sheets.
    *   **Run:** `python scripts/6_..._Analyze_Human_Eval...py`
    *   **Input:** Prompts for the directory containing completed rater CSVs (from Step 6). Prompts for the keyed sheet CSV (from Step 5). Prompts for an output directory.
    *   **Output:** Summary statistics CSV, IAA scores CSV, qualitative examples CSV, logs.

## 8. Results Summary

*   **Model Comparison:** BART-large significantly outperformed T5-small on ROUGE metrics.
*   **Final BART-large Test Performance:** Achieved **ROUGE-L: 0.3460** on the test set.
*   **Human Evaluation:** Showed moderate rater agreement (Kappa ~0.4-0.5). Model-generated subjects were rated as generally good (Avg scores > 3.1/4), but lower than human-written subjects (Avg scores ~3.5-3.8), especially on conciseness.
*   **Generated Length:** The best model produced a median subject length of 22 tokens on the test set, longer than the human average (~4).

*For detailed results and analysis, please refer to the full project report.*

## 9. Demo (Streamlit App)

An interactive demo application is provided in `scripts/app_...py`.

**To Run the Demo Locally:**

1.  **Update Model Path:** Edit the `MODEL_PATH` variable inside the Streamlit script (`scripts/app_...py`) to point to the directory containing your downloaded best fine-tuned BART-large model checkpoint (from Step 4).
2.  **Run:** Execute the following command in your terminal (ensure you are in the project's root directory and your virtual environment is active):
    ```bash
    streamlit run scripts/app_...py
    ```
3.  Open the provided local URL (e.g., `http://localhost:8501`) in your web browser.

**(Optional: Add a GIF/Screenshot of the running app here)**


## 10. License

All Rights Reserved - see LICENSE file for more.

## 11. Acknowledgements

*   Dataset: Annotated Enron Subject Line Corpus (AESLC) by Zhang and Tetreault.
*   Libraries: Hugging Face (Transformers, Datasets, Evaluate), PyTorch, Pandas, NLTK, Streamlit, Scikit-learn.
*   Compute: Google Colab.

---
