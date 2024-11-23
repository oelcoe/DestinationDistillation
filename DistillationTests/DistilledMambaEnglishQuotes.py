import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from torch.nn import KLDivLoss
import tensorflow as tf
device = torch.device("cpu")
# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

# Define the model and dataset
MODEL_ID = "state-spaces/mamba-130m-hf"
DATASET_NAME = "Abirate/english_quotes"
SAVE_DIR = "./distilled_mamba_english_quotes"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
teacher_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

# Load and preprocess dataset
print("Loading and preprocessing dataset...")
dataset = load_dataset(DATASET_NAME, split="train")
subset_dataset = dataset.select(range(10))  # Use a smaller subset for training

def preprocess_function(examples):
    return tokenizer(examples["quote"], padding="max_length", truncation=True)

subset_dataset = subset_dataset.map(preprocess_function, batched=True)

# Prepare data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Create a smaller student model
print("Creating smaller student model...")
config = AutoConfig.from_pretrained(MODEL_ID)
config.num_hidden_layers = config.num_hidden_layers // 8  # Reduce the number of layers
student_model = AutoModelForCausalLM.from_config(config)

# Define distillation loss
kl_loss = KLDivLoss(reduction="sum")

def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """
    Computes the distillation loss (KL-divergence) between teacher and student logits.

    Args:
        student_logits: Logits produced by the student model.
        teacher_logits: Logits produced by the teacher model.
        temperature: Temperature scaling parameter for distillation.

    Returns:
        Distillation loss value.
    """
    # Scale logits by temperature
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    # Compute soft targets from teacher logits
    teacher_probs = torch.nn.functional.softmax(teacher_logits, dim=-1)

    # Compute KL Divergence loss
    loss = kl_loss(torch.nn.functional.log_softmax(student_logits, dim=-1), teacher_probs)
    return loss / student_logits.size(0)  # Normalize by batch size

# Custom Trainer for Distillation
class DistillationTrainer(Trainer):
    """
    Custom Trainer for distillation, where the student model learns from the teacher model's outputs.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch = None):
        # Ensure inputs are on the correct device
        inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
        
        # Get teacher logits
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        # Get student logits
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        # Compute distillation loss
        loss = distillation_loss(student_logits, teacher_logits)
        return (loss, student_outputs) if return_outputs else loss
student_model.gradient_checkpointing_enable()

training_args = TrainingArguments(
    output_dir=SAVE_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=1,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=50,
    bf16=False,  # Automatically use BF16 if supported
    load_best_model_at_end=True,
    save_total_limit=2,
)


# Initialize and start training
print("Starting training...")
trainer = DistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=subset_dataset,
    eval_dataset=subset_dataset,  # Use the same subset for evaluation
    tokenizer=tokenizer,
    data_collator=data_collator,
)
teacher_model.to(training_args.device)
trainer.train()

# Save the student model and tokenizer
print("Saving distilled model...")
student_model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

# Evaluation and testing
print("Evaluating the distilled model...")

def evaluate_model(model, dataset, tokenizer):
    """
    Evaluates the model on the provided dataset and prints sample outputs.

    Args:
        model: Trained model to evaluate.
        dataset: Dataset to evaluate on.
        tokenizer: Tokenizer for encoding/decoding.
    """
    input_text = "Be yourself; everyone else is already taken."
    input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]

    print("\nGenerating text with the trained model:")
    generated_ids = model.generate(input_ids, max_new_tokens=10)
    print(f"Input: {input_text}")
    print("Output:", tokenizer.decode(generated_ids[0], skip_special_tokens=True))

# Evaluate and test the student model
evaluate_model(student_model, subset_dataset, tokenizer)