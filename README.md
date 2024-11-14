# NLPII_Proyect

#### Model Selection:
Choose a pre-trained model that balances performance with memory efficiency. Suggested models include LLama-3.1-8B, Mistral 7B v0.3, and Qwen2.5-7B.

#### Dataset Selection:
Select a task-specific, high-quality dataset to maximize training effectiveness with limited data. Options include Databricks Dolly 15K, OpenAssistant Conversations Dataset, and LIMA. Ensure datasets are aligned with the model's task requirements.

#### Training Scheme:
Decide on a mix of fine-tuning, alignment (e.g., Reinforcement Learning from Human Feedback or Direct Preference Optimization), model compression (e.g., pruning, quantization), and hyperparameters to stay within memory and time limits.
Use tools like Optuna for hyperparameter optimization and calculate memory usage based on the number of trainable parameters.

#### Model Evaluation and Checkpointing:
Set regular checkpoints to save model states and avoid data loss due to interruptions. For final model evaluation, use IFEval and other latency benchmarks to assess performance.
