# src/modelos_transformers.py
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset
import numpy as np

class DatasetMedico(Dataset):
    def __init__(self, textos, etiquetas, tokenizer, max_length=256):
        self.textos = textos
        self.etiquetas = etiquetas
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.textos)
    
    def __getitem__(self, idx):
        texto = str(self.textos[idx])
        etiquetas = self.etiquetas[idx]
        
        encoding = self.tokenizer(
            texto,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(etiquetas, dtype=torch.float)
        }

class ModeloTransformer:
    def __init__(self, nombre_modelo='bert-base-uncased', num_etiquetas=4):
        self.tokenizer = AutoTokenizer.from_pretrained(nombre_modelo)
        self.modelo = AutoModel.from_pretrained(nombre_modelo)
        self.clasificador = torch.nn.Linear(self.modelo.config.hidden_size, num_etiquetas)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.modelo = self.modelo.to(self.device)
        self.clasificador = self.clasificador.to(self.device)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.modelo(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.clasificador(pooled_output)
    
    def predecir(self, textos, umbral=0.5):
        self.modelo.eval()
        predicciones = []
        
        with torch.no_grad():
            for texto in textos:
                encoding = self.tokenizer(
                    texto,
                    truncation=True,
                    padding='max_length',
                    max_length=256,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                outputs = self.forward(input_ids, attention_mask)
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > umbral).astype(int)
                predicciones.append(preds[0])
        
        return np.array(predicciones)