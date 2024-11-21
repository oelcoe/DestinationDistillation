
from mamba.model import MambaTextClassification
from MiniMamba.student import MambaStudent
from dataset import ImdbDataset
from utils import compute_metrics

from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments
from MiniMamba.distillation import DistillationTrainer


# token = os.getenv("HUGGINGFACE_TOKEN") ---nn
# login(token=token, write_permission=True)  ---- nn

# ------- Load Teacher and Student

# How do we load our pretrained Mamba model?????
teacher_model = MambaTextClassification.from_pretrained("state-spaces/mamba-130m")
teacher_model.to("cuda")

# Student 
 student_model = MambaStudent.from_small_config(
        device='cuda', 
        num_classes=2,  # Binary classification
        d_model=256,    # Adjust to fine-tune parameter count
        n_layers=4
    )

# Count parameters
count_parameters(model) # we are aiming for 10% of the parameters 


# ----------- Import Tokenizer
# Load the tokenizer of the Mamba model from the gpt-neox-20b model.
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
# Set the pad token id to the eos token id in the tokenizer.
tokenizer.pad_token_id = tokenizer.eos_token_id


# ----------- Load Data Set
imdb = load_dataset("imdb")
imdbDataset = ImdbDataset(imdb, tokenizer)
train_dataset = imdbDataset.return_train_dataset()
test_dataset, eval_dataset = imdbDataset.return_test_dataset(eval_ratio=0.1)


training_args = TrainingArguments(
    output_dir="./distilled_mamba",
    learning_rate=3e-5,                # Learning rate
    num_train_epochs=3,                # Train for 3 epochs
    per_device_train_batch_size=16,    # Smaller model can handle larger batch sizes
    per_device_eval_batch_size=64,     # Faster evaluation
    gradient_accumulation_steps=2,     # Adjust effective batch size
    evaluation_strategy="epoch",       # Evaluate after every epoch
    logging_strategy="steps",          # Determine when to log information
    logging_steps=100,                 # Log training stats every 100 steps
    save_strategy="epoch",             # Save model after each epoch
    save_steps=0.1,                    # Number of steps between saving checkpoints
    fp16=True,                         # Use mixed precision for faster training
    logging_dir="./logs",              # Save logs for TensorBoard or other tools
    load_best_model_at_end=True,       # Load the model with the best evaluation result during training
)

trainer = DistillationTrainer(
    model=student_model,  
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=subset_dataset,
    eval_dataset=subset_dataset,
    data_collator=data_collator,
)

trainer.train()