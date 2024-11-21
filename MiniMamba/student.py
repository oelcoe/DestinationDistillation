import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

class SmallMambaConfig:
    def __init__(
        self, 
        d_model=256,    # Increased model dimension
        n_layers=4,     # Increased number of layers
        vocab_size=50257,  # Standard GPT-2 tokenizer vocab size
        d_state=16,     # State dimension
        expand_factor=2,  # Expansion factor
        dt_rank=16,     # Rank for dt projection
        num_classes=2   # Binary classification
    ):
        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.d_state = d_state
        self.expand_factor = expand_factor
        self.dt_rank = dt_rank
        self.num_classes = num_classes

class MambaStudent(MambaLMHeadModel):
    def __init__(
        self,
        config: SmallMambaConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        # Convert SmallMambaConfig to a dictionary for MambaLMHeadModel
        mamba_config_dict = {
            'd_model': config.d_model,
            'n_layers': config.n_layers,
            'vocab_size': config.vocab_size,
            'd_state': config.d_state,
            'expand_factor': config.expand_factor,
            'dt_rank': config.dt_rank
        }
        
        super().__init__(mamba_config_dict, initializer_cfg, device, dtype)
        
        # Create a classification head using a linear layer
        self.classification_head = nn.Linear(config.d_model, config.num_classes)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Pass input_ids through the backbone model to receive hidden_states
        hidden_states = self.backbone(input_ids)
        
        # Take the mean of hidden_states along the second dimension
        mean_hidden_states = hidden_states.mean(dim=1)
        
        # Pass mean_hidden_states through the classification head to get logits
        logits = self.classification_head(mean_hidden_states)
        
        # Language modeling head remains intact
        lm_logits = self.lm_head(hidden_states)
        
        if labels is None:
            ClassificationOutput = namedtuple("ClassificationOutput", ["logits", "lm_logits"])
            return ClassificationOutput(logits=logits, lm_logits=lm_logits)
        else:
            # Assuming you might want to handle both classification and language modeling loss
            ClassificationOutput = namedtuple("ClassificationOutput", ["loss", "logits", "lm_logits"])
            
            # Classification loss
            cls_loss_fct = nn.CrossEntropyLoss()
            cls_loss = cls_loss_fct(logits, labels)
            
            # Optional: If you want to compute language modeling loss as well
            # This would require passing lm_labels
            lm_labels = None  # You'd pass this if you want LM loss
            lm_loss = None
            if lm_labels is not None:
                lm_loss_fct = nn.CrossEntropyLoss()
                lm_loss = lm_loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            
            return ClassificationOutput(
                loss=cls_loss, 
                logits=logits, 
                lm_logits=lm_logits
            )
    
    def predict(self, text, tokenizer, id2label=None):
        input_ids = torch.tensor(tokenizer(text)['input_ids'], device="cuda")[None]
        with torch.no_grad():
            output = self.forward(input_ids)
            logits = output.logits[0]
            label = np.argmax(logits.cpu().numpy())
            
        if id2label is not None:
            return id2label[label]
        else:
            return label
    
    @classmethod
    def from_small_config(cls, device=None, dtype=None, **kwargs):
        # Create a small Mamba configuration
        config = SmallMambaConfig(**kwargs)
        
        # Initialize the model from the configuration
        model = cls(config, device=device, dtype=dtype)
        
        return model.to(device)

# Function to count model parameters
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    return total_params

