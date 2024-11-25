import os
import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    TrainingArguments,
    Trainer,
    get_scheduler
)
from datasets import load_dataset

# Constants
MODEL_ID = "state-spaces/mamba-130m-hf"
DATASET_NAME = "sst2"  # Changed to SST-2
NUM_LABELS = 2  # Changed to 2 for binary classification
SAVE_DIR = "./distilled_mamba_sst2"

# Label mapping for SST-2
SENTIMENT_LABELS = {
    0: "negative",
    1: "positive"
}

# Device setup
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("cpu")
    else:
        return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

class MambaForSequenceClassification(nn.Module):
    def __init__(self, base_model, num_labels, pooling_type='mean', freeze_base=False):  # Added freeze_base parameter
        super().__init__()
        self.mamba = base_model
        self.num_labels = num_labels
        self.pooling_type = pooling_type

        # Only freeze layers if specified (for teacher model)
        if freeze_base:
            total_layers = len(list(base_model.parameters()))
            for param in list(base_model.parameters())[:(total_layers - 2)]:
                param.requires_grad = False
            print("- Freezing all but last 2 layers of base model")
        else:
            print("- All layers trainable")

        # Get hidden size from model config
        self.hidden_size = self.mamba.config.hidden_size

        # Create projection layer
        # Create projection layer
        self.projection = nn.Sequential(
           nn.Identity()
        )

        # Classification head
        self.classifier = nn.Sequential(
            # nn.Linear(self.hidden_size, self.hidden_size // 2),
            # nn.GELU(),
            # nn.Dropout(0.1),
            # nn.Linear(self.hidden_size // 2, num_labels)
            # nn.Dropout(0.3),
            # nn.Linear(self.hidden_size, num_labels)

            nn.Linear(self.mamba.config.vocab_size, num_labels),
            nn.Dropout(0.1)
        )

        print(f"\nModel initialized with:")
        print(f"- Hidden size: {self.hidden_size}")
        print(f"- Vocab size: {self.mamba.config.vocab_size}")
        print(f"- Number of layers: {self.mamba.config.num_hidden_layers}")

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get Mamba output
        outputs = self.mamba(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.logits

        # Apply pooling
        if self.pooling_type == 'mean':
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                pooled_output = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
            else:
                pooled_output = hidden_states.mean(dim=1)
        else:  # max pooling
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                hidden_states = hidden_states * mask_expanded - 1e9 * (1 - mask_expanded)
            pooled_output = hidden_states.max(dim=1)[0]

        # Project and classify
        projected = self.projection(pooled_output)
        logits = self.classifier(projected)

        # Compute loss if needed
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return type('MambaSequenceClassifierOutput', (), {
            'loss': loss,
            'logits': logits
        })

    @property
    def device(self):
        return next(self.parameters()).device

    def to(self, device):
        super().to(device)
        self.mamba = self.mamba.to(device)
        self.projection = self.projection.to(device)
        self.classifier = self.classifier.to(device)
        return self

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def create_dataset(dataset_name, tokenizer, split="train", max_length=64, use_subset=True):
    """Create a properly formatted dataset."""
    # Load dataset - use full eval set but subset of train
    if split == "validation" or not use_subset:  # SST-2 uses 'validation' instead of 'test'
        dataset = load_dataset(dataset_name, split=split)
    else:
        # Only use subset for training data
        dataset = load_dataset(dataset_name, split=f"{split}[:1%]")

    print(f"\nLoading {split} dataset:")
    print(f"Number of examples: {len(dataset)}")

    # Tokenize texts
    encodings = tokenizer(
        dataset["sentence"],  # SST-2 uses 'sentence' instead of 'text'
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors=None
    )

    # Get labels
    labels = dataset["label"]

    return CustomDataset(encodings, labels)

class OptimizedDistillationTrainer:
    def __init__(
        self,
        teacher_model,
        student_model,
        train_dataloader,
        eval_dataloader,
        num_epochs,
        device,
        num_labels,
        project_name="mamba-distillation",  # Added for wandb
        experiment_name="sst2",
        learning_rate=1e-4,
        weight_decay=0.01,
        gradient_accumulation_steps=1
    ):
        self.teacher_model = teacher_model.to(device)
        self.student_model = student_model.to(device)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.num_epochs = num_epochs
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_labels = num_labels
        # Initialize wandb
        wandb.init(
            project=project_name,
            name=experiment_name,
            config={
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "num_epochs": num_epochs,
                "batch_size": train_dataloader.batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "teacher_params": sum(p.numel() for p in teacher_model.parameters()),
                "student_params": sum(p.numel() for p in student_model.parameters()),
                "model_reduction": f"{sum(p.numel() for p in student_model.parameters()) / sum(p.numel() for p in teacher_model.parameters()):.2%}"
            }
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            student_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        num_training_steps = len(train_dataloader) * num_epochs
        self.scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=num_training_steps // 10,
            num_training_steps=num_training_steps
        )

    def compute_loss(self, student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
        """Compute the distillation loss."""
        # Temperature scaling
        student_scaled = student_logits / temperature
        teacher_scaled = teacher_logits / temperature

        # Compute soft targets once
        soft_targets = F.softmax(teacher_scaled, dim=-1)
        student_log_probs = F.log_softmax(student_scaled, dim=-1)

        # Compute losses
        soft_loss = -(soft_targets * student_log_probs).sum(dim=-1).mean()
        hard_loss = F.cross_entropy(student_logits, labels)

        return alpha * (temperature ** 2) * soft_loss + (1 - alpha) * hard_loss

    def train(self):
        best_eval_acc = 0
        for epoch in range(self.num_epochs):
            self.student_model.train()
            self.teacher_model.eval()

            total_loss = 0
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")

            for step, batch in enumerate(progress_bar):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Get teacher predictions
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(**batch)

                # Get student predictions and loss
                student_outputs = self.student_model(**batch)
                loss = self.compute_loss(
                    student_outputs.logits,
                    teacher_outputs.logits,
                    batch['labels']
                ) / self.gradient_accumulation_steps

                # Backward pass
                loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                total_loss += loss.item() * self.gradient_accumulation_steps
                current_loss = total_loss / (step + 1)
                current_lr = self.scheduler.get_last_lr()[0]

                # Update progress bar and log to wandb
                progress_bar.set_postfix({
                    'loss': f"{current_loss:.4f}",
                    'lr': f"{current_lr:.2e}"
                })

                # Log training metrics
                wandb.log({
                    "train/loss": current_loss,
                    "train/learning_rate": current_lr,
                    "train/step": epoch * len(self.train_dataloader) + step
                })

            # Evaluation
            eval_metrics = self.evaluate()
            print(f"Epoch {epoch+1} - Eval Accuracy: {eval_metrics['accuracy']:.4f}")

            # Log evaluation metrics
            wandb.log({
                "eval/accuracy": eval_metrics["accuracy"],
                "eval/teacher_accuracy": eval_metrics["teacher_accuracy"],
                "eval/accuracy_retention": eval_metrics["accuracy_retention"],
                "eval/epoch": epoch
            })

            # Save best model
            if eval_metrics['accuracy'] > best_eval_acc:
                best_eval_acc = eval_metrics['accuracy']
                self.save_model()

                # Log best metrics
                wandb.log({
                    "best/accuracy": best_eval_acc,
                    "best/epoch": epoch
                })

        # Close wandb run
        wandb.finish()

    def evaluate(self):
        self.student_model.eval()
        self.teacher_model.eval()

        student_correct = 0
        teacher_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Teacher predictions
                teacher_outputs = self.teacher_model(**batch)
                teacher_preds = teacher_outputs.logits.argmax(dim=-1)

                # Student predictions
                student_outputs = self.student_model(**batch)
                student_preds = student_outputs.logits.argmax(dim=-1)

                # Calculate accuracy
                student_correct += (student_preds == batch['labels']).sum().item()
                teacher_correct += (teacher_preds == batch['labels']).sum().item()
                total_samples += batch['labels'].size(0)

        student_accuracy = student_correct / total_samples
        teacher_accuracy = teacher_correct / total_samples
        accuracy_retention = (student_accuracy / teacher_accuracy) * 100 if teacher_accuracy > 0 else 0

        return {
            "accuracy": student_accuracy,
            "teacher_accuracy": teacher_accuracy,
            "accuracy_retention": accuracy_retention
        }

    def save_model(self):
        os.makedirs(SAVE_DIR, exist_ok=True)
        torch.save(self.student_model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pth'))

class TeacherTrainer:
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader,
        num_epochs,
        device,
        learning_rate=5e-4,  # Lower learning rate for fine-tuning
        weight_decay=0.01
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.num_epochs = num_epochs
        self.device = device

        # Separate parameter groups for fine-tuning
        # Higher learning rate for new layers, lower for pre-trained layers
        classifier_params = list(model.classifier.parameters()) + list(model.projection.parameters())
        pretrained_params = list(model.mamba.parameters())[-2:]  # Last two layers of base model

        self.optimizer = torch.optim.AdamW([
            {'params': classifier_params, 'lr': learning_rate},
            {'params': pretrained_params, 'lr': learning_rate * 0.1}  # Lower learning rate for pre-trained layers
        ], weight_decay=weight_decay)

        # Learning rate scheduler with warm-up
        num_training_steps = len(train_dataloader) * num_epochs
        num_warmup_steps = num_training_steps // 10

        self.scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    def train(self):
        best_eval_acc = 0
        early_stopping_patience = 3
        early_stopping_counter = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(self.train_dataloader, desc=f"Teacher Fine-tuning Epoch {epoch+1}")

            for step, batch in enumerate(progress_bar):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss

                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()
                progress_bar.set_postfix({
                    'loss': total_loss / (step + 1),
                    'lr': self.scheduler.get_last_lr()[0]
                })

            # Evaluation
            eval_acc = self.evaluate()
            print(f"Teacher Epoch {epoch+1} - Eval Accuracy: {eval_acc:.4f}")

            # Early stopping check
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                early_stopping_counter = 0
                self.save_model()
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered!")
                break

        print(f"Best Teacher Accuracy: {best_eval_acc:.4f}")
        return best_eval_acc

    def evaluate(self):
        self.model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                predictions = outputs.logits.argmax(dim=-1)

                total_correct += (predictions == batch['labels']).sum().item()
                total_samples += batch['labels'].size(0)
        return total_correct / total_samples

    def save_model(self):
        os.makedirs(SAVE_DIR, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(SAVE_DIR, 'best_teacher.pth'))

def count_parameters(model):
    """Count trainable and total parameters in a model."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    # Count parameters by component
    mamba_params = sum(p.numel() for p in model.mamba.parameters())
    projection_params = sum(p.numel() for p in model.projection.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())

    size_mb = total_params * 4 / (1024 * 1024)  # Size in MB (assuming float32)

    return {
        'trainable': trainable_params,
        'total': total_params,
        'mamba_base': mamba_params,
        'projection': projection_params,
        'classifier': classifier_params,
        'size_mb': size_mb
    }

def compare_models(teacher_model, student_model):
    """Compare and display model sizes."""
    teacher_params = count_parameters(teacher_model)
    student_params = count_parameters(student_model)

    print("\nModel Size Comparison:")
    print("\nTeacher Model:")
    print(f"Total Parameters: {teacher_params['total']:,}")
    print(f"Trainable Parameters: {teacher_params['trainable']:,}")
    print(f"Model Size: {teacher_params['size_mb']:.2f} MB")
    print("\nParameter distribution:")
    print(f"- Mamba base: {teacher_params['mamba_base']:,}")
    print(f"- Projection layer: {teacher_params['projection']:,}")
    print(f"- Classifier: {teacher_params['classifier']:,}")

    print("\nStudent Model:")
    print(f"Total Parameters: {student_params['total']:,}")
    print(f"Trainable Parameters: {student_params['trainable']:,}")
    print(f"Model Size: {student_params['size_mb']:.2f} MB")
    print("\nParameter distribution:")
    print(f"- Mamba base: {student_params['mamba_base']:,}")
    print(f"- Projection layer: {student_params['projection']:,}")
    print(f"- Classifier: {student_params['classifier']:,}")

    reduction_ratio = teacher_params['total'] / student_params['total']
    size_reduction = (1 - student_params['total'] / teacher_params['total']) * 100

    print("\nReduction Statistics:")
    print(f"Reduction Ratio: {reduction_ratio:.2f}x")
    print(f"Size Reduction: {size_reduction:.2f}%")

def create_reduced_mamba_config(teacher_config):
    """Create a properly reduced configuration for the student model."""
    student_config = AutoConfig.from_pretrained(MODEL_ID)

    # Reduce all relevant dimensions
    student_config.hidden_size = teacher_config.hidden_size // 4  # Hidden size
    student_config.num_hidden_layers = max(1, teacher_config.num_hidden_layers // 12)  # Number of layers
    # student_config.d_state = teacher_config.d_state // 2        # State dimension
    # student_config.d_conv = max(4, teacher_config.d_conv // 2)  # Conv dimension
    # student_config.expand = max(1, teacher_config.expand // 2)  # Expansion factor

    return student_config

def perform_final_evaluation(teacher_model, student_model, eval_dataloader, device):
    """Perform comprehensive evaluation of both models on the entire eval set."""
    teacher_model.eval()
    student_model.eval()

    teacher_correct = 0
    student_correct = 0
    total_samples = 0

    # Track predictions for both models
    teacher_predictions = {i: 0 for i in SENTIMENT_LABELS.keys()}
    student_predictions = {i: 0 for i in SENTIMENT_LABELS.keys()}
    true_labels_dist = {i: 0 for i in SENTIMENT_LABELS.keys()}

    # Track confusion matrices
    teacher_confusion = {i: {j: 0 for j in SENTIMENT_LABELS.keys()} for i in SENTIMENT_LABELS.keys()}
    student_confusion = {i: {j: 0 for j in SENTIMENT_LABELS.keys()} for i in SENTIMENT_LABELS.keys()}

    print("\nPerforming final evaluation...")
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['labels']

            teacher_outputs = teacher_model(**batch)
            teacher_preds = teacher_outputs.logits.argmax(dim=-1)

            student_outputs = student_model(**batch)
            student_preds = student_outputs.logits.argmax(dim=-1)

            teacher_correct += (teacher_preds == labels).sum().item()
            student_correct += (student_preds == labels).sum().item()
            total_samples += labels.size(0)

            for pred in teacher_preds.cpu().numpy():
                teacher_predictions[pred] += 1
            for pred in student_preds.cpu().numpy():
                student_predictions[pred] += 1
            for label in labels.cpu().numpy():
                true_labels_dist[label] += 1

            for true, t_pred, s_pred in zip(labels.cpu().numpy(),
                                          teacher_preds.cpu().numpy(),
                                          student_preds.cpu().numpy()):
                teacher_confusion[true][t_pred] += 1
                student_confusion[true][s_pred] += 1

    # Calculate metrics
    teacher_accuracy = teacher_correct / total_samples
    student_accuracy = student_correct / total_samples

    # Print results
    print("\n=== Final Evaluation Results ===")
    print("\nOverall Accuracy:")
    print(f"Teacher Model: {teacher_accuracy:.4f}")
    print(f"Student Model: {student_accuracy:.4f}")
    print(f"Accuracy Retention: {(student_accuracy/teacher_accuracy)*100:.2f}%")

    print("\nTrue Label Distribution:")
    for label_id, count in true_labels_dist.items():
        percentage = (count / total_samples) * 100
        print(f"{SENTIMENT_LABELS[label_id]}: {count} samples ({percentage:.2f}%)")

    print("\nTeacher Model Predictions:")
    for label_id, count in teacher_predictions.items():
        percentage = (count / total_samples) * 100
        print(f"{SENTIMENT_LABELS[label_id]}: {count} predictions ({percentage:.2f}%)")

    print("\nStudent Model Predictions:")
    for label_id, count in student_predictions.items():
        percentage = (count / total_samples) * 100
        print(f"{SENTIMENT_LABELS[label_id]}: {count} predictions ({percentage:.2f}%)")

    print("\nPer-Class Performance:")
    for sentiment_id in SENTIMENT_LABELS.keys():
        sentiment_name = SENTIMENT_LABELS[sentiment_id]
        true_count = true_labels_dist[sentiment_id]
        if true_count == 0:
            continue

        teacher_correct = teacher_confusion[sentiment_id][sentiment_id]
        student_correct = student_confusion[sentiment_id][sentiment_id]

        print(f"\n{sentiment_name}:")
        print(f"Teacher accuracy: {teacher_correct/true_count:.4f}")
        print(f"Student accuracy: {student_correct/true_count:.4f}")


def main():
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Create datasets
    print("Creating datasets...")
    train_dataset = create_dataset(DATASET_NAME, tokenizer, split="train", use_subset=True)
    eval_dataset = create_dataset(DATASET_NAME, tokenizer, split="validation", use_subset=True)

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2
    )

    # Create teacher model
    print("Creating teacher model...")
    base_teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    teacher_model = MambaForSequenceClassification(base_teacher, NUM_LABELS, freeze_base=True)
    teacher_model = teacher_model.to(device)

    # Print teacher model's configuration
    print("\nTeacher Model Configuration:")
    print(f"hidden_size: {base_teacher.config.hidden_size}")
    print(f"num_hidden_layers: {base_teacher.config.num_hidden_layers}")
    # print(f"d_state: {base_teacher.config.d_state}")
    # print(f"d_conv: {base_teacher.config.d_conv}")
    # print(f"expand: {base_teacher.config.expand}")

    # Create properly reduced student model
    print("\nCreating student model...")
    student_config = create_reduced_mamba_config(base_teacher.config)
    base_student = AutoModelForCausalLM.from_config(student_config)
    student_model = MambaForSequenceClassification(base_student, NUM_LABELS)
    student_model = student_model.to(device)
    # Print student model's configuration
    print("\nStudent Model Configuration:")
    print(f"hidden_size: {base_student.config.hidden_size}")
    print(f"num_hidden_layers: {base_student.config.num_hidden_layers}")
    # print(f"d_state: {base_student.config.d_state}")
    # print(f"d_conv: {base_student.config.d_conv}")
    # print(f"expand: {base_student.config.expand}")

    # Compare model sizes
    compare_models(teacher_model, student_model)

    def count_trainable_params(model, name):
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"\n{name} parameter counts:")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Total parameters: {total:,}")
        print(f"Percentage trainable: {(trainable/total)*100:.2f}%")

    count_trainable_params(teacher_model, "Teacher model")
    count_trainable_params(student_model, "Student model")

    # Fine-tune teacher model
    print("\nFine-tuning teacher model...")
    teacher_trainer = TeacherTrainer(
        model=teacher_model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        num_epochs=10,  # More epochs for fine-tuning
        device=device
    )

    #best_teacher_acc = teacher_trainer.train()

    #best_teacher_acc = teacher_trainer.train()
    #print(f"\nTeacher training completed. Best accuracy: {best_teacher_acc:.4f}")

    # Load best teacher model
    teacher_model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_teacher.pth')))
    teacher_model.eval()  # Set to evaluation mode

    # Initialize distillation trainer
    print("\nStarting distillation...")
    distillation_trainer = OptimizedDistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        num_epochs=3,
        device=device,
        num_labels=NUM_LABELS,
        project_name="mamba-distillation",
        experiment_name=f"sst2-student-{student_config.num_hidden_layers}layers",
        learning_rate=1e-4,
        gradient_accumulation_steps=2
    )

    # Train student through distillation
    distillation_trainer.train()

    test_texts = [
        "This movie was fantastic!",
        "What a terrible waste of time.",
        "A masterpiece of modern cinema.",
        "I really didn't enjoy this film at all."
    ]

    print("\nTesting both models:")
    for text in test_texts:
        print(f"\nText: {text}")
        teacher_pred, teacher_probs = predict_text(text, teacher_model, tokenizer, device)
        student_pred, student_probs = predict_text(text, student_model, tokenizer, device)

        print(f"Teacher prediction: {SENTIMENT_LABELS[teacher_pred]} (confidence: {teacher_probs[teacher_pred]:.4f})")
        print(f"Student prediction: {SENTIMENT_LABELS[student_pred]} (confidence: {student_probs[student_pred]:.4f})")

    print("\nPerforming final comprehensive evaluation...")
    perform_final_evaluation(teacher_model, student_model, eval_dataloader, device)

def predict_text(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1)

    return prediction.item(), probs[0].cpu().numpy()

if __name__ == "__main__":
    main()