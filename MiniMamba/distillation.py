from transformers import Trainer
from torch.nn import KLDivLoss
import torch

# Distillation loss function
kl_loss = KLDivLoss(reduction="batchmean")

def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    teacher_probs = torch.nn.functional.softmax(teacher_logits, dim=-1)
    loss = kl_loss(torch.nn.functional.log_softmax(student_logits, dim=-1), teacher_probs)
    return loss

class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Get teacher model outputs
        with torch.no_grad():
            teacher_outputs = teacher_model(**{k: v.to(self.args.device) for k, v in inputs.items()})
            teacher_logits = teacher_outputs.logits

        # Get student model outputs
        student_outputs = model(**{k: v.to(self.args.device) for k, v in inputs.items()})
        student_logits = student_outputs.logits

        # Compute distillation loss
        loss = distillation_loss(student_logits, teacher_logits)
        return (loss, student_outputs) if return_outputs else loss
