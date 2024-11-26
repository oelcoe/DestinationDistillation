import os
import torch
import torch.nn as nn
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
from tqdm.auto import tqdm
import psutil
import random
from datetime import datetime
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import defaultdict
import numpy as np
import torch.nn.functional as F
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import warnings


MAX_TRAIN_SAMPLES = 1  # Adjust this number for your desired dataset size
MAX_EVAL_SAMPLES = 100   # Adjust evaluation set size
TRAIN_BATCH_SIZE = 32     # Adjust batch sizes
EVAL_BATCH_SIZE = 64

# Constants
MODEL_ID = "state-spaces/mamba-130m-hf"
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-103-v1"
SAVE_DIR = "./distilled_mamba_wikitext"
MAX_LENGTH = 128

# Device setup
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

def print_sample_texts(dataset, num_samples=3):
    """Print random samples from the dataset for quality checking."""
    print(f"\nSample texts from dataset:")
    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        text = dataset[idx]['text']
        print(f"\nSample {i+1} (length: {len(text)}):")
        print("-" * 50)
        print(text[:200] + "..." if len(text) > 200 else text)
        print("-" * 50)

def process_and_check_sample(tokenizer, text):
    """Process a single text sample and return quality metrics."""
    # Tokenize
    tokens = tokenizer.encode(text)
    
    # Calculate metrics
    metrics = {
        'length': len(tokens),
        'unique_tokens': len(set(tokens)),
        'token_diversity': len(set(tokens)) / len(tokens) if tokens else 0,
    }
    
    return metrics

def analyze_dataset_quality(dataset, tokenizer, num_samples=100):
    """Analyze dataset quality metrics."""
    print("\nAnalyzing dataset quality...")
    metrics = []
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    for idx in tqdm(indices, desc="Analyzing samples"):
        text = dataset[idx]['text']
        sample_metrics = process_and_check_sample(tokenizer, text)
        metrics.append(sample_metrics)
    
    # Calculate aggregate statistics
    avg_length = np.mean([m['length'] for m in metrics])
    avg_diversity = np.mean([m['token_diversity'] for m in metrics])
    
    print(f"\nDataset Quality Metrics:")
    print(f"Average sequence length: {avg_length:.2f}")
    print(f"Average token diversity: {avg_diversity:.2f}")



def load_and_preprocess_dataset(tokenizer):
    """Load and preprocess the WikiText dataset with size control."""
    print(f"\nLoading {DATASET_NAME} dataset...")
    
    # Load dataset with size limits
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    eval_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="validation")
    
    # Take subset of data
    if MAX_TRAIN_SAMPLES:
        dataset = dataset.select(range(min(len(dataset), MAX_TRAIN_SAMPLES)))
    if MAX_EVAL_SAMPLES:
        eval_dataset = eval_dataset.select(range(min(len(eval_dataset), MAX_EVAL_SAMPLES)))
    
    print(f"\nDataset statistics:")
    print(f"Training examples: {len(dataset):,}")
    print(f"Validation examples: {len(eval_dataset):,}")
    
    def preprocess_function(examples):
        """Tokenize and prepare for causal language modeling."""
        tokenized = tokenizer(
            examples["text"],
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors=None,
            return_attention_mask=True,
            return_special_tokens_mask=True
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        for i, mask in enumerate(tokenized["attention_mask"]):
            tokenized["labels"][i] = [
                label if mask_val == 1 else -100
                for label, mask_val in zip(tokenized["labels"][i], mask)
            ]
        
        return tokenized
    
    # Process datasets with progress bars
    print("\nProcessing training dataset...")
    processed_train = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Processing train data",
    )
    
    print("\nProcessing validation dataset...")
    processed_eval = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Processing validation data",
    )
    
    return processed_train, processed_eval


def setup_nltk():
    """Download required NLTK data with error handling."""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        return True
    except Exception as e:
        print(f"Warning: Failed to download NLTK data: {e}")
        return False

class GenerationEvaluator:
    def __init__(self, teacher_model, student_model, tokenizer, device):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.tokenizer = tokenizer
        self.device = device
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1
        self.nltk_available = setup_nltk()
    
    def tokenize_safely(self, text):
        """Safely tokenize text with fallback options."""
        if self.nltk_available:
            try:
                return nltk.word_tokenize(text)
            except Exception as e:
                warnings.warn(f"NLTK tokenization failed: {e}")
        
        # Fallback to simple tokenization
        return text.split()
    
    def evaluate_continuation(self, prompt, max_length=50, num_samples=5):
        """Evaluate text continuation capabilities with error handling."""
        metrics = defaultdict(list)
        
        # Generate multiple samples from both models
        teacher_generations = self.generate_samples(self.teacher_model, prompt, max_length, num_samples)
        student_generations = self.generate_samples(self.student_model, prompt, max_length, num_samples)
        
        # Calculate metrics for each sample pair
        for teacher_gen, student_gen in zip(teacher_generations, student_generations):
            # ROUGE scores
            rouge_scores = self.rouge_scorer.score(teacher_gen, student_gen)
            for key, score in rouge_scores.items():
                metrics[f'rouge_{key}_f1'].append(score.fmeasure)
            
            # BLEU score with safe tokenization
            teacher_tokens = self.tokenize_safely(teacher_gen)
            student_tokens = self.tokenize_safely(student_gen)
            
            try:
                bleu_score = sentence_bleu([teacher_tokens], student_tokens, smoothing_function=self.smoothing)
                metrics['bleu'].append(bleu_score)
            except Exception as e:
                warnings.warn(f"BLEU score calculation failed: {e}")
                metrics['bleu'].append(0.0)
            
            # Token probability analysis
            try:
                teacher_probs = self.calculate_token_probabilities(self.teacher_model, prompt, teacher_gen)
                student_probs = self.calculate_token_probabilities(self.student_model, prompt, student_gen)
                
                metrics['teacher_mean_prob'].append(np.mean(teacher_probs) if teacher_probs else 0)
                metrics['student_mean_prob'].append(np.mean(student_probs) if student_probs else 0)
            except Exception as e:
                warnings.warn(f"Probability calculation failed: {e}")
                metrics['teacher_mean_prob'].append(0)
                metrics['student_mean_prob'].append(0)
            
            # Perplexity with error handling
            try:
                metrics['teacher_perplexity'].append(self.calculate_perplexity(self.teacher_model, teacher_gen))
                metrics['student_perplexity'].append(self.calculate_perplexity(self.student_model, student_gen))
            except Exception as e:
                warnings.warn(f"Perplexity calculation failed: {e}")
                metrics['teacher_perplexity'].append(float('inf'))
                metrics['student_perplexity'].append(float('inf'))   
            
            # Length similarity
            length_ratio = len(student_gen) / max(len(teacher_gen), 1)
            metrics['length_ratio'].append(length_ratio)
        
        # Aggregate metrics, handling potential empty lists
        return {k: np.mean(v) if v else 0.0 for k, v in metrics.items()}
        
    def generate_samples(self, model, prompt, max_length, num_samples):
        """Generate multiple samples from a model with proper attention mask."""
        generations = []
        # Properly tokenize with attention mask
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        for _ in range(num_samples):
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generations.append(generated_text)
        
        return generations

    def calculate_token_probabilities(self, model, input_text, generated_text):
        """Calculate token-by-token probability distributions with attention mask."""
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_attention_mask=True
        ).to(self.device)
        
        target_ids = self.tokenizer(
            generated_text, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_attention_mask=True
        )["input_ids"].to(self.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            
            # Get probability for each generated token
            token_probs = []
            for i in range(min(logits.shape[1], target_ids.shape[1])):
                if i < logits.shape[1]:
                    prob = probs[0, i, target_ids[0, i]].item()
                    token_probs.append(prob)
                        
        return token_probs

def calculate_perplexity(model, tokenizer, text, device):
    """Calculate perplexity with proper handling of padding and attention masks."""
    try:
        # Tokenize with attention mask
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_attention_mask=True
        ).to(device)
        
        # Create shifted labels for causal language modeling
        labels = inputs["input_ids"].clone()
        
        # Calculate loss with proper masking
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss  # This is already averaged over non-padded tokens
            
        return torch.exp(loss).item()
    except Exception as e:
        print(f"Perplexity calculation error: {e}")
        return float('nan')
    
    def evaluate_continuation(self, prompt, max_length=50, num_samples=5):
        """Evaluate text continuation capabilities."""
        metrics = defaultdict(list)
        
        # Generate multiple samples from both models
        teacher_generations = self.generate_samples(self.teacher_model, prompt, max_length, num_samples)
        student_generations = self.generate_samples(self.student_model, prompt, max_length, num_samples)
        
        # Calculate metrics for each sample pair
        for teacher_gen, student_gen in zip(teacher_generations, student_generations):
            # ROUGE scores
            rouge_scores = self.rouge_scorer.score(teacher_gen, student_gen)
            for key, score in rouge_scores.items():
                metrics[f'rouge_{key}_f1'].append(score.fmeasure)
            
            # BLEU score
            teacher_tokens = nltk.word_tokenize(teacher_gen)
            student_tokens = nltk.word_tokenize(student_gen)
            bleu_score = sentence_bleu([teacher_tokens], student_tokens, smoothing_function=self.smoothing)
            metrics['bleu'].append(bleu_score)
            
            # Token probability analysis
            teacher_probs = self.calculate_token_probabilities(self.teacher_model, prompt, teacher_gen)
            student_probs = self.calculate_token_probabilities(self.student_model, prompt, student_gen)
            
            metrics['teacher_mean_prob'].append(np.mean(teacher_probs))
            metrics['student_mean_prob'].append(np.mean(student_probs))
            
            # Perplexity
            metrics['teacher_perplexity'].append(self.calculate_perplexity(self.teacher_model, teacher_gen))
            metrics['student_perplexity'].append(self.calculate_perplexity(self.student_model, student_gen))
            
            # Length similarity
            length_ratio = len(student_gen) / len(teacher_gen)
            metrics['length_ratio'].append(length_ratio)
        
        # Aggregate metrics
        return {k: np.mean(v) for k, v in metrics.items()}

def evaluate_generation_quality(teacher_model, student_model, tokenizer, device):
    """Generation quality evaluation with fixed perplexity."""
    test_prompts = [
        "The future of technology depends on",
        "In the field of artificial intelligence,",
        "The development of new methods for",
    ]
    
    print("\n=== Generation Quality Evaluation ===")
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        
        # Generate from both models
        input_ids = tokenizer(prompt, return_tensors="pt", padding=True, return_attention_mask=True)
        input_ids = {k: v.to(device) for k, v in input_ids.items()}
        
        # Teacher generation
        with torch.no_grad():
            teacher_output = teacher_model.generate(
                **input_ids,
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                top_k=50,
                top_p=0.95,  # Add nucleus sampling
                no_repeat_ngram_size=3  # Prevent repetition
            )
            teacher_text = tokenizer.decode(teacher_output[0], skip_special_tokens=True)
        
        # Student generation
        with torch.no_grad():
            student_output = student_model.generate(
                **input_ids,
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                top_k=50,
                top_p=0.95,
                no_repeat_ngram_size=3
            )
            student_text = tokenizer.decode(student_output[0], skip_special_tokens=True)
        
        # Print generations
        print("\nTeacher generated:")
        print(teacher_text)
        print("\nStudent generated:")
        print(student_text)
        
        # Calculate metrics
        teacher_perplexity = calculate_perplexity(teacher_model, tokenizer, teacher_text, device)
        student_perplexity = calculate_perplexity(student_model, tokenizer, student_text, device)
        
        # Basic text quality metrics
        teacher_words = teacher_text.split()
        student_words = student_text.split()
        
        # Calculate repetition rate
        def get_repetition_rate(words):
            if not words:
                return 0
            return 1 - (len(set(words)) / len(words))
        
        teacher_repetition = get_repetition_rate(teacher_words)
        student_repetition = get_repetition_rate(student_words)
        
        print("\nMetrics:")
        print(f"Length (words): Teacher={len(teacher_words)}, Student={len(student_words)}")
        print(f"Unique words: Teacher={len(set(teacher_words))}, Student={len(set(student_words))}")
        print(f"Repetition rate: Teacher={teacher_repetition:.2f}, Student={student_repetition:.2f}")
        print(f"Perplexity: Teacher={teacher_perplexity:.2f}, Student={student_perplexity:.2f}")
        
        # Also look at generation diversity
        print("\nGeneration diversity check...")
        diverse_generations = []
        for _ in range(3):
            with torch.no_grad():
                output = student_model.generate(
                    **input_ids,
                    max_length=50,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95
                )
                text = tokenizer.decode(output[0], skip_special_tokens=True)
                diverse_generations.append(text)
        
        print("\nMultiple generations from student model:")
        for i, gen in enumerate(diverse_generations, 1):
            print(f"\nGeneration {i}:")
            print(gen)

def main():
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load and process dataset
    train_dataset, eval_dataset = load_and_preprocess_dataset(tokenizer)
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Load teacher model
    print("\nLoading teacher model...")
    teacher_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    teacher_model.to(device)
    
    # Create student model
    print("\nCreating student model...")
    student_config = AutoConfig.from_pretrained(MODEL_ID)
    student_config.hidden_size = student_config.hidden_size // 4
    student_config.num_hidden_layers = max(1, student_config.num_hidden_layers // 4)
    student_model = AutoModelForCausalLM.from_config(student_config)
    student_model.to(device)
    
    # Print model sizes
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    print(f"\nModel sizes:")
    print(f"Teacher parameters: {teacher_params:,}")
    print(f"Student parameters: {student_params:,}")
    print(f"Size reduction: {(1 - student_params/teacher_params)*100:.1f}%")

    training_args = TrainingArguments(
        output_dir=SAVE_DIR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=4,  # Adjust this based on memory usage
        learning_rate=1e-4,
        num_train_epochs=3,
        warmup_steps=100,  # Reduced warmup steps for smaller dataset
        evaluation_strategy="steps",
        eval_steps=50,     # More frequent evaluation for smaller dataset
        save_steps=100,    # More frequent saving for smaller dataset
        logging_dir="./logs",
        logging_steps=10,  # More frequent logging
        bf16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        save_total_limit=2,
        remove_unused_columns=False,
        # Memory optimization options
        gradient_checkpointing=True,
        fp16_full_eval=True,
        dataloader_num_workers=0,  # Set to higher number if you have more CPU cores
        dataloader_pin_memory=True
    )

    # Optional: Add memory monitoring
    def print_memory_usage():
        if torch.cuda.is_available():
            print(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            print(f"GPU Memory cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
        print(f"CPU Memory usage: {psutil.Process().memory_info().rss/1e9:.2f} GB")

    # Modified trainer initialization with memory monitoring
    class MonitoredDistillationTrainer(DistillationTrainer):
        def training_step(self, *args, **kwargs):
            if self.state.global_step % 100 == 0:  # Monitor every 100 steps
                print_memory_usage()
            return super().training_step(*args, **kwargs)

    trainer = MonitoredDistillationTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        teacher_model=teacher_model
    )
    print("\nStarting training...")
    trainer.train()
    
    # Save the distilled model
    print("\nSaving distilled model...")
    student_model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    
    # Final evaluation
    print("\nPerforming comprehensive generation evaluation...")
    evaluate_generation_quality(teacher_model, student_model, tokenizer, device)

class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
        labels = labels.to(self.args.device)
        
        # Make sure teacher model is on the right device
        if self.teacher_model.device != self.args.device:
            self.teacher_model = self.teacher_model.to(self.args.device)
        
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        
        student_outputs = model(**inputs)
        
        loss = distillation_loss(
            student_outputs.logits,
            teacher_outputs.logits,
            labels,
            temperature=2.0,
            alpha=0.5
        )
        
        inputs["labels"] = labels
        
        return (loss, student_outputs) if return_outputs else loss

def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    """Compute distillation loss with soft and hard targets."""
    label_mask = (labels != -100).unsqueeze(-1)
    
    scaled_student = student_logits / temperature
    scaled_teacher = teacher_logits.detach() / temperature
    
    soft_targets = torch.nn.functional.softmax(scaled_teacher, dim=-1)
    log_probs_student = torch.nn.functional.log_softmax(scaled_student, dim=-1)
    
    soft_loss = -(soft_targets * log_probs_student).sum(dim=-1)
    soft_loss = (soft_loss * label_mask.squeeze(-1)).mean()
    
    hard_loss = torch.nn.functional.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )
    
    return alpha * (temperature ** 2) * soft_loss + (1 - alpha) * hard_loss

if __name__ == "__main__":
    main()