
# Llama 3.1 Fine-Tuning for Instruction-based tasks
## NLP II Final Project - Rust'eze

Welcome to the **NLPII_Project**, an advanced natural language processing project designed to fine-tune a pretrained model such as Llama 3.1 to improve performance in instruction-based tasks. This repository contains the code, datasets, and documentation necessary to replicate our experiments and explore the power of NLP techniques.

## Table of Contents
1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [How to Run](#how-to-run)
4. [Fine-Tuning Configuration](#fine-tuning-configuration)

---

## Installation

### Prerequisites
Ensure the following software is installed:
- Python 3.8 or higher
- Git
- Get a HuggingFace Token by registering at https://huggingface.co/ and creating a new token
- Get Llama 3.1 access token by fulfilling the form at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct 

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/HenryECA/NLPII_Proyect.git
   cd NLPII_Proyect
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a ``keys_file.py`` and include your token as TOKEN (preferably the Llama 3.1 Access Token):
   ```bash
   -- keys_file.py --

   TOKEN = "Include your token"
   ```
---

## Project Structure

```
NLPII_Proyect/
│
├── data/                 # Dataset folder
│   ├── train.csv         # Training dataset
│   ├── test.csv          # Testing dataset
│   └── sample_text.txt   # Example input for inference
│
├── models/               # Pre-trained and saved models
├── scripts/              # Additional utility scripts
├── config.json           # Configuration file
├── train.py              # Script to train the model
├── evaluate.py           # Script for evaluation
├── predict.py            # Script for inference
├── main.ipynb            # Jupyter notebook for advanced configuration and experimentation
├── requirements.txt      # Python dependencies
└── README.md             # Documentation
```

---

## How to Run

1. **Execute ``main.ipynb`` with our best model:**
   
   Run the `main.ipynb` script with a configuration file:
   ```bash
   <!-- python main.ipynb  -->
   ```

   This will start the training schema (which you can avoid if you do not need to train and just want to evaluate).
   If you indeed want to train a new model, you can change the model parameters inside the "Configurations" and "Training" sections in ``main.ipyn``. 
   During this process, a test prompt will be tried before and after the model fine-tuning. This will help the user know objectivelly how well the model performs without chacking the ifeval metrics.


2. **Evaluate Performance - IfEval:**

   First, browse to the ``/IfEval`` directory.
   Then, run `nlp_ifeval.ipynb` to get the IfEval responses with your best model. Remember to change the path to it. Also, if you want to see the metrics, just execute the command given in the ifeval_command.txt file on the temrinal. 

   Finally, the metrics will be presented printed in the terminal to check the model's performance in several different scenarios.
   
---

## Fine-Tuning Configuration

Advanced fine-tuning techniques, such as configuring LoRA (Low-Rank Adaptation) and other fine-tuning methods, are supported in this project. To customize these configurations:

1. Open the `main.ipynb` notebook in a Jupyter environment.
2. Follow the step-by-step guide in the notebook to:
   - Adjust specific parameters.
   - Experiment with various fine-tuning approaches tailored to your dataset and model.
   - Visualize and log performance metrics during fine-tuning.



---
