import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from tqdm import tqdm

# Define the model and dataset
MODEL_ID = "state-spaces/mamba-130m-hf"
DATASET_NAME = "Abirate/english_quotes"
SAVE_DIR = "./distilled_mamba_english_quotes"


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Load tokenizer and model
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Set padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

teacher_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
teacher_model.to(device)

# Load and preprocess dataset
print("Loading and preprocessing dataset...")
dataset = load_dataset(DATASET_NAME, split="train")
subset_dataset = dataset#.select(range(10))

def preprocess_function(examples):
    """
    Tokenize the texts and prepare for causal language modeling.
    """
    # Tokenize the texts
    tokenized = tokenizer(
        examples["quote"],
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors=None,
        return_attention_mask=True,
        return_special_tokens_mask=True
    )
    
    # Create labels for causal language modeling (shift inputs right)
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    # For any padding tokens in the labels, replace with -100 so they're ignored in the loss
    for i, mask in enumerate(tokenized["attention_mask"]):
        tokenized["labels"][i] = [
            label if mask_val == 1 else -100
            for label, mask_val in zip(tokenized["labels"][i], mask)
        ]
    
    return tokenized

# Process the dataset
processed_dataset = subset_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=subset_dataset.column_names,
    desc="Tokenizing texts"
)

# Create evaluation dataset with the same preprocessing
eval_dataset = processed_dataset

# Use DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're doing causal language modeling
)

# Create student model
print("Creating student model...")
config = AutoConfig.from_pretrained(MODEL_ID)

# Modify config for smaller model
config.num_hidden_layers = max(1, config.num_hidden_layers // 8)

# Create student model with modified config
student_model = AutoModelForCausalLM.from_config(config)
student_model.to(device)

def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    """
    Enhanced distillation loss combining soft and hard targets.
    
    Args:
        student_logits: Logits from student model (batch_size, seq_len, vocab_size)
        teacher_logits: Logits from teacher model (batch_size, seq_len, vocab_size)
        labels: Ground truth labels (batch_size, seq_len)
        temperature: Temperature for softmax
        alpha: Weight for soft targets loss
    """
    # Create mask for valid positions (exclude padding/ignored positions)
    label_mask = (labels != -100).unsqueeze(-1)  # Add vocab dimension
    
    # Scale logits by temperature
    scaled_student = student_logits / temperature
    scaled_teacher = teacher_logits.detach() / temperature
    
    # Compute soft targets (KL divergence)
    soft_targets = torch.nn.functional.softmax(scaled_teacher, dim=-1)
    log_probs_student = torch.nn.functional.log_softmax(scaled_student, dim=-1)
    
    soft_loss = -(soft_targets * log_probs_student).sum(dim=-1)  # Sum over vocabulary
    soft_loss = (soft_loss * label_mask.squeeze(-1)).mean()  # Average over non-padded positions
    
    # Compute hard targets (cross entropy)
    hard_loss = torch.nn.functional.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100  # Ignore padded positions
    )
    
    # Combine losses
    return alpha * (temperature ** 2) * soft_loss + (1 - alpha) * hard_loss

class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Get labels and remove them from inputs for model forward pass
        labels = inputs.pop("labels")
        
        # Ensure inputs are on the correct device
        inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
        labels = labels.to(self.args.device)
        
        # Get teacher predictions
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)
        
        # Get student predictions
        student_outputs = model(**inputs)
        
        # Compute distillation loss
        loss = distillation_loss(
            student_outputs.logits,
            teacher_outputs.logits,
            labels,
            temperature=2.0,
            alpha=0.5
        )
        
        # Add labels back to inputs for potential future use
        inputs["labels"] = labels
        
        return (loss, student_outputs) if return_outputs else loss

# Enable gradient checkpointing
student_model.gradient_checkpointing_enable()

# Training arguments
training_args = TrainingArguments(
    output_dir=SAVE_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    num_train_epochs=3,
    warmup_steps=100,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=50,
    bf16=torch.cuda.is_available(),
    load_best_model_at_end=True,
    save_total_limit=2,
    remove_unused_columns=False
)

# Initialize trainer and train
print("Starting training...")
trainer = DistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=processed_dataset,
    eval_dataset=processed_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

teacher_model.to(training_args.device)
trainer.train()

# Save the distilled model
print("Saving distilled model...")
student_model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

# Load tokenizer and model
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Set padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

teacher_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

# Load and preprocess dataset
print("Loading and preprocessing dataset...")
dataset = load_dataset(DATASET_NAME, split="train")
subset_dataset = dataset.select(range(10))

def preprocess_function(examples):
    """
    Tokenize the texts and prepare for causal language modeling.
    """
    # Tokenize the texts
    tokenized = tokenizer(
        examples["quote"],
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors=None,
        return_attention_mask=True,
        return_special_tokens_mask=True
    )
    
    # Create labels for causal language modeling (shift inputs right)
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    # For any padding tokens in the labels, replace with -100 so they're ignored in the loss
    for i, mask in enumerate(tokenized["attention_mask"]):
        tokenized["labels"][i] = [
            label if mask_val == 1 else -100
            for label, mask_val in zip(tokenized["labels"][i], mask)
        ]
    
    return tokenized

# Process the datasets
processed_dataset = subset_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=subset_dataset.column_names,
    desc="Tokenizing texts"
)

# Create evaluation dataset with the same preprocessing
eval_dataset = processed_dataset

# Use DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're doing causal language modeling
)

class ModelEvaluator:
    def __init__(self, teacher_model, student_model, tokenizer, dataset, device, data_collator):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.device = device
        self.data_collator = data_collator
        
        # Move models to the appropriate device
        self.teacher_model.to(device)
        self.student_model.to(device)
        # Set models to evaluation mode
        self.teacher_model.eval()
        self.student_model.eval()
        # Initialize metrics dictionary
        self.metrics = {}

    def compute_perplexity(self, model, input_ids, attention_mask, labels):
        """Compute perplexity for a batch of inputs."""
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        model.to(device)
        print(f"Teacher model device: {model.device}")
        print(f"Attention device: {attention_mask.device}")
        print(f"Input tensor device: {input_ids.device}")
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return torch.exp(outputs.loss).item()

    def compute_activation_similarity(self, teacher_activations, student_activations):
        """Compute CKA similarity between activations."""
        def center(K):
            n = K.shape[0]
            unit = torch.ones([n, n], device=K.device)
            I = torch.eye(n, device=K.device)
            H = I - unit / n
            return torch.matmul(torch.matmul(H, K), H)

        def compute_gram(x):
            return torch.matmul(x, x.t())

        gram_teacher = compute_gram(teacher_activations)
        gram_student = compute_gram(student_activations)
        
        centered_teacher = center(gram_teacher)
        centered_student = center(gram_student)
        
        normalized_hs = torch.norm(centered_student) * torch.norm(centered_teacher)
        return torch.trace(torch.matmul(centered_teacher, centered_student)) / normalized_hs

    def get_hidden_states(self, model, input_ids, attention_mask):
        """Extract hidden states from the model."""
        with torch.no_grad():
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            return outputs.hidden_states

    def evaluate_sequence_prediction(self, num_samples=100):
        """Evaluate sequence prediction accuracy."""
        predictions = {'teacher': [], 'student': []}
        ground_truth = []
        
        eval_dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self.data_collator
        )
        
        for i, batch in enumerate(eval_dataloader):
            if i >= num_samples:
                break
                
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Generate predictions
            with torch.no_grad():
                teacher_model.to(self.device)
                teacher_output = self.teacher_model.generate(
                    batch['input_ids'], 
                    max_new_tokens=10,
                    num_return_sequences=1,
                    attention_mask=batch['attention_mask']
                )
                student_model.to(self.device)
                student_output = self.student_model.generate(
                    batch['input_ids'],
                    max_new_tokens=10,
                    num_return_sequences=1,
                    attention_mask=batch['attention_mask']
                )
            
            # Decode predictions
            predictions['teacher'].append(
                self.tokenizer.decode(teacher_output[0], skip_special_tokens=True)
            )
            predictions['student'].append(
                self.tokenizer.decode(student_output[0], skip_special_tokens=True)
            )
            ground_truth.append(
                self.tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)
            )
            
        return predictions, ground_truth

    def compute_metrics(self, eval_dataset):
        """Compute comprehensive evaluation metrics."""
        all_teacher_perplexity = []
        all_student_perplexity = []
        all_activation_similarities = []
        
        # Create dataloader with data collator
        dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=self.data_collator
        )
        
        for batch in tqdm(dataloader, desc="Computing metrics"):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Compute perplexity
            teacher_ppl = self.compute_perplexity(
                self.teacher_model, 
                batch['input_ids'], 
                batch['attention_mask'],
                batch['labels']
            )
            student_ppl = self.compute_perplexity(
                self.student_model, 
                batch['input_ids'], 
                batch['attention_mask'],
                batch['labels']
            )
            
            all_teacher_perplexity.append(teacher_ppl)
            all_student_perplexity.append(student_ppl)
            
            # Get hidden states for activation similarity
            teacher_hidden_states = self.get_hidden_states(
                self.teacher_model, 
                batch['input_ids'], 
                batch['attention_mask']
            )
            student_hidden_states = self.get_hidden_states(
                self.student_model, 
                batch['input_ids'], 
                batch['attention_mask']
            )
            
            # Compare final layer activations
            teacher_final = teacher_hidden_states[-1].view(batch['input_ids'].size(0), -1)
            student_final = student_hidden_states[-1].view(batch['input_ids'].size(0), -1)
            
            similarity = self.compute_activation_similarity(teacher_final, student_final)
            all_activation_similarities.append(similarity.item())
        
        # Compute sequence prediction metrics
        predictions, ground_truth = self.evaluate_sequence_prediction()
        
        self.metrics = {
            'teacher_perplexity': np.mean(all_teacher_perplexity),
            'student_perplexity': np.mean(all_student_perplexity),
            'activation_similarity': np.mean(all_activation_similarities),
            'perplexity_ratio': np.mean(all_student_perplexity) / np.mean(all_teacher_perplexity),
            'predictions': predictions,
            'ground_truth': ground_truth
        }
        
        return self.metrics

def evaluate_and_log_metrics(teacher_model, student_model, tokenizer, eval_dataset, device, data_collator):
    """Main evaluation function that computes and logs all metrics."""
    evaluator = ModelEvaluator(
        teacher_model, 
        student_model, 
        tokenizer, 
        eval_dataset, 
        device,
        data_collator
    )
    
    metrics = evaluator.compute_metrics(eval_dataset)
    
    # Log metrics
    print("\nEvaluation Metrics:")
    print(f"Teacher Perplexity: {metrics['teacher_perplexity']:.2f}")
    print(f"Student Perplexity: {metrics['student_perplexity']:.2f}")
    print(f"Activation Similarity: {metrics['activation_similarity']:.2f}")
    print(f"Perplexity Ratio (Student/Teacher): {metrics['perplexity_ratio']:.2f}")
    
    return metrics

# Evaluation TODO: Debug this
eval_metrics = evaluate_and_log_metrics(
    teacher_model,
    student_model,
    tokenizer,
    eval_dataset,  # Use the processed evaluation dataset
    training_args.device,
    data_collator
)

input_ids = tokenizer("Hello, who is this?", return_tensors= "pt")["input_ids"]
input_ids = input_ids.to(device)
student_model.to(device)
teacher_model.to(device)
out = teacher_model.generate(input_ids, max_new_tokens=20)
print("Original Model")
print(tokenizer.batch_decode(out))
out = student_model.generate(input_ids, max_new_tokens=20)
print("Distilled Model")
print(tokenizer.batch_decode(out))