import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
import evaluate
import numpy as np
from torch.nn import KLDivLoss
from torch.nn.functional import softmax, log_softmax

# Load SST-2 dataset
sst2 = load_dataset("sst2")

# Preprocess the dataset
tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True)

tokenized_sst2 = sst2.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Reduce dataset size to 10% for faster training
train_dataset = tokenized_sst2["train"].select(range(len(tokenized_sst2["train"])))
validation_dataset = tokenized_sst2["validation"].select(range(len(tokenized_sst2["validation"])))

# Define compute metrics
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load teacher model
teacher_model = AutoModelForSequenceClassification.from_pretrained(
    "./transformers_teacher"
).to(device)

# Define training arguments for teacher model
training_args = TrainingArguments(
    output_dir="./transformers_teacher",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

# Train the teacher model
trainer = Trainer(
    model=teacher_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
#trainer.train()
trainer.evaluate()

# Save the teacher model
teacher_model.save_pretrained("./transformers_teacher")
tokenizer.save_pretrained("./transformers_teacher")

# Create a smaller student model
small_config = teacher_model.config
small_config.hidden_size //= 2  # Reduce hidden size
small_config.num_attention_heads //= 2  # Reduce attention heads
student_model = AutoModelForSequenceClassification.from_config(small_config).to(device)

# Distillation loss function
kl_loss = KLDivLoss(reduction="batchmean")

def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    student_probs = log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = softmax(teacher_logits / temperature, dim=-1)
    return kl_loss(student_probs, teacher_probs)

# Define distillation trainer
class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Get student logits
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        # Get teacher logits
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        # Compute distillation loss
        loss = distillation_loss(student_logits, teacher_logits)
        return (loss, student_outputs) if return_outputs else loss

# Define training arguments for student model
distillation_args = TrainingArguments(
    output_dir="./transformers_student",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

# Train the student model
distillation_trainer = DistillationTrainer(
    model=student_model,
    args=distillation_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

distillation_trainer.train()
distillation_trainer.evaluate()

trainer_student = Trainer(
    model=teacher_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
# Save the student model
student_model.save_pretrained("./transformers_student")
tokenizer.save_pretrained("./transformers_student")

# Evaluate Teacher and Student Models
def evaluate_model(model, dataset, description):
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    results = trainer.evaluate()
    print(f"{description} Accuracy: {results['eval_accuracy']:.4f}")

# Evaluate Teacher
evaluate_model(teacher_model, validation_dataset, "Teacher Model")

# Evaluate Student
evaluate_model(student_model, validation_dataset, "Student Model")