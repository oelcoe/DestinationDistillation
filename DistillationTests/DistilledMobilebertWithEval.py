import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
import numpy as np
from torch.nn import KLDivLoss, CrossEntropyLoss
from torch.nn.functional import log_softmax, softmax
import evaluate
from tqdm import tqdm

# Define metrics and datasets configuration
metrics = {
    "accuracy": evaluate.load("accuracy"),
    "f1": evaluate.load("f1"),
    "pearson": evaluate.load("pearsonr"),
}

# Configuration for datasets and models
model_dataset_configs = {
    "sst2": {
        "model_name": "google/mobilebert-uncased",
        "dataset_name": "sst2",
        "metrics": "accuracy",
    },
    "mnli": {
        "model_name": "google/mobilebert-uncased",
        "dataset_name": "mnli",
        "metrics": "accuracy",
    },
    # Add more datasets/models here as needed
}

def compute_metrics(eval_pred, metric_key):
    """
    Computes evaluation metrics such as accuracy, F1 score, or Pearson correlation.

    Args:
        eval_pred: Tuple containing model predictions and ground truth labels.
        metric_key: Key to select the metric from the global `metrics` dictionary.

    Returns:
        Dictionary with computed metric value.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    metric = metrics[metric_key]
    return metric.compute(predictions=predictions, references=labels)

def combined_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    """
    Computes the combined loss: distillation loss (KL-divergence) + cross-entropy loss.

    Args:
        student_logits: Logits produced by the student model.
        teacher_logits: Logits produced by the teacher model.
        labels: Ground truth labels.
        temperature: Temperature scaling parameter for distillation.
        alpha: Weight for the distillation loss (1 - alpha for cross-entropy loss).

    Returns:
        Combined loss value.
    """
    # Compute KL-divergence (distillation loss)
    student_probs = log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = softmax(teacher_logits / temperature, dim=-1)
    distillation_loss = KLDivLoss(reduction="batchmean")(student_probs, teacher_probs)

    # Compute cross-entropy loss
    cross_entropy_loss = CrossEntropyLoss()(student_logits, labels)

    # Combine losses
    return alpha * distillation_loss + (1 - alpha) * cross_entropy_loss

class DistillationTrainer(Trainer):
    """
    Custom Trainer for distillation, where the student model learns from the teacher model's outputs.

    Args:
        teacher_model: Pretrained teacher model used for distillation.
    """
    def __init__(self, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Overrides the default compute_loss to include combined distillation and cross-entropy loss.

        Args:
            model: The student model being trained.
            inputs: Input data batch.
            return_outputs: Whether to return model outputs along with the loss.

        Returns:
            Combined loss (and outputs if return_outputs is True).
        """
        # Get student logits
        labels = inputs.pop("labels")
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        # Get teacher logits
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        # Compute combined loss
        loss = combined_loss(student_logits, teacher_logits, labels)
        return (loss, student_outputs) if return_outputs else loss

def main(args):
    """
    Main function to train the teacher model, distill it into a student model,
    and evaluate both models on the selected dataset.

    Args:
        args: Command-line arguments parsed by argparse.
    """
    # Load dataset configuration
    config = model_dataset_configs[args.dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset and tokenizer
    dataset = load_dataset("glue", config["dataset_name"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # Preprocessing function for tokenization
    def preprocess_function(examples):
        return tokenizer(examples["sentence"], truncation=True, max_length=128)

    # Tokenize the dataset
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer)

    # Reduce dataset size for faster training (optional)
    train_dataset = tokenized_dataset["train"].select(range(len(tokenized_dataset["train"]) // 10))
    validation_dataset = tokenized_dataset["validation"].select(range(len(tokenized_dataset["validation"]) // 10))

    # Load and train the teacher model
    teacher_model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], num_labels=2
    ).to(device)
    teacher_args = TrainingArguments(
        output_dir="./teacher_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir="./logs_teacher",
    )
    teacher_trainer = Trainer(
        model=teacher_model,
        args=teacher_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, config["metrics"]),
    )
    print("Training Teacher Model...")
    teacher_trainer.train()
    print("Evaluating Teacher Model...")
    teacher_results = teacher_trainer.evaluate()
    print(f"Teacher {config['metrics']}:", teacher_results[f"eval_{config['metrics']}"])

    # Save the trained teacher model
    teacher_model.save_pretrained("./teacher_model")

    # Create and train the student model
    student_config = teacher_model.config
    student_config.hidden_size //= 2  # Reduce hidden size
    student_config.num_attention_heads //= 2  # Reduce attention heads
    student_model = AutoModelForSequenceClassification(student_config).to(device)
    student_args = TrainingArguments(
        output_dir="./student_model",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        num_train_epochs=2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir="./logs_student",
    )
    student_trainer = DistillationTrainer(
        teacher_model=teacher_model,
        model=student_model,
        args=student_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, config["metrics"]),
    )
    print("Training Student Model...")
    student_trainer.train()
    print("Evaluating Student Model...")
    student_results = student_trainer.evaluate()
    print(f"Student {config['metrics']}:", student_results[f"eval_{config['metrics']}"])

    # Save the trained student model
    student_model.save_pretrained("./student_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and distill MobileBERT on various datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=model_dataset_configs.keys(),
        help="Dataset to use (e.g., sst2, cola).",
    )
    args = parser.parse_args()
    main(args)