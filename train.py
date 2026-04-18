'''basato su nanogpt di Karpathy ma con le seguenti modifiche:

'''

import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F 
# -------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd deve essere divisibile per n_head"
        #creo le matrici QKV per l'attenzione, 3 * n_embd per concatenarle
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        #creo la matrice di output dell'attenzione
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        #uso il tag "SCALE_INIT" per indicare che questa matrice deve essere inizializzata con una deviazione standard più piccola per stabilizzare l'addestramento
        self.c_proj.SCALE_INIT = 1
        #regolarizzazione
        self.n_head = config.n_head
        self.n_embd = config.n_embd
    
    def forward(self, x):
        B, T, C = x.size()
        #calcolo le matrici QKV da c_attn
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2) #divido in 3 matrici di dimensione n_embd nella dimensione colonne
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # BxTx(n_head)xdimensione testa -> Bx(n_head)xTxdimensione testa
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # BxTx(n_head)xdimensione testa -> Bx(n_head)xTxdimensione testa
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # BxTx(n_head)xdimensione testa -> Bx(n_head)xTxdimensione testa
        #flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        #riporto la matrice alla dimensione originale
        y = y.transpose(1, 2).contiguous().view(B,T,C)
        #la passo alla matrice dell'output dell'attenzione
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # SwiGLU: usiamo hidden_dim = 8/3 * n_embd per mantenere lo stesso numero di parametri di GELU con 4x
        hidden_dim = int(config.n_embd * 8 / 3)
        # c_fc produce due proiezioni in parallelo: una per il valore, una per il gate
        self.c_fc   = nn.Linear(config.n_embd, 2 * hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.c_proj.SCALE_INIT = 1

    def forward(self, x):
        # split in valore e gate
        x, gate = self.c_fc(x).chunk(2, dim=-1)
        # SwiGLU: valore * Swish(gate)
        x = x * F.silu(gate)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class MyModelConfig:
    block_size: int = 1024
    vocab_size: int = 100277 #100000 BPE merges, 256 bytes tokens, 21 per i token specializ
    n_embd: int = 768 #dimensione embedding 
    n_layers: int = 12 #numero di blocchi
    n_head: int = 12 #numero di teste di attenzione
    # dropout:float = 0.0 #dropout rate

class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self. config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range (config.n_layers)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # rendo la matrice dei pesi condivisa tra i token embeddings e il language model head
        self.transformer.wte.weight = self.lm_head.weight
        
        #inizializzo i pesi del modello, apply chiama la funzione _init_weight su ogni modulo del modello
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'SCALE_INIT'):
                #per le matrici con SCALE_INIT, uso la deviazione calcolata con la formula di Xavier, 1/sqrt(fan_in) dove fanin è il numero di input della matrice
                std *= (2 * self.config.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, idx, targets=None):
        #size di idx: B (batch size) x T (block size)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward the sequence of {T} size because the block size in the model is only {self.config.block_size}"
        #creo la matrice di posizioni TxT
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        #prendo i token e position embeddings
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        #forward pass sei blocchi del modello
        for block in self.transformer.h:
            x = block(x)
        #ultimo layer norm e language model head
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        #separo i parametri sogetti a weight decay e quelli che non lo sono
        params = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        params = {pn: p for pn, p in params.items() if p.requires_grad}
        #prendo i tensori 2D per quelli che ricevono il weight decay e quelli non
        decay = [p for n, p in params.items() if p.dim() >= 2]
        no_decay = [p for n, p in params.items() if p.dim() < 2]
        #creo i gruppi per l'ootimizer
        optim_groups = [
            {'params':decay, 'weight_decay': weight_decay},
            {'params':no_decay, 'weight_decay': 0.0}
        ]
        #uso la fused AdamW optimizer di pytorch se è disponibile
        fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused and device_type == 'cuda'
        if use_fused:
            print('using fused AdamW')
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9,0.999), eps=1e-8, fused=use_fused)
        return optimizer
#--------------------
import tiktoken
import numpy as np

def load_tokens(filename):
    np_tokens = np.load(filename).astype(np.int32)
    t_tokens = torch.tensor(np_tokens, dtype=torch.long)
    return t_tokens

class DataLoader():
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}, "split must be 'train' or 'val'"
        
        root = 'edu_fineweb10B' #da fineweb-edu.py
        shards = os.listdir(root)
        shards = [s for s in shards if split in s] #i file sono chiamati f"edufineweb_{split}_{shard_index:06d}"
        shards = [os.path.join(root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"No shards found for split {split} in {root}"
        self.reset()
    
    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buffer = self.tokens[self.current_position : self.current_position + B * T + 1] #prendo B*T+1 token per avere B sequenze di input e target
        x = buffer[:-1].view(B, T) #input è tutto tranne l'ultimo token
        y = buffer[1:].view(B, T) #target è tutto tranne il primo token
        self.current_position += B * T #incremento la posizione del buffer
        #controllo se siamo alla fine dello shard, se sì carico il prossimo shard
        if self.current_position + B * T >= len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards) #uso un approccio circolare per le shards
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0
        return x, y

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
device_type = 'cuda' if device.type.startswith('cuda') else 'cpu'
print(f"device: {device_type}")

enc = tiktoken.get_encoding("cl100k_base")

total_batch_size = 2**19
B = 64
T = 1024
assert total_batch_size % (B * T) == 0, "total_batch_size must be divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f'B={B}, T={T}, grad_accum_steps={grad_accum_steps}')

train_loader = DataLoader(B, T, split='train')
val_loader = DataLoader(B, T, split='val')

torch.set_float32_matmul_precision('high')

model = MyModel(MyModelConfig()).to(device)
raw_model = model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
num_epochs = 1
max_steps = num_epochs * 19073 #19073 viene calcolato con il numero di token totali (10B) diviso il total_batch_size (2**19)
#funzione per calcolare il learning rate dell'iterazione i-esima
def get_lr(step):
    #warm up lineare fino a max_lr per i primi warmup_steps
    if step < warmup_steps:
        return max_lr * (step+1) / warmup_steps
    #dopo i max_step il learning rate è min_lr
    if step > max_steps:
        return min_lr
    #decay con cosinusoide da max_lr a min_lr nell'intervallo tra warmup_steps e max_steps
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1, "decay_ratio must be in [0,1]"
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device_type=device_type)

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'log.txt')
with open(log_file, 'w') as f:
    pass #creo il file di log vuoto

for step in range(max_steps):
    t0 = time.time() #uso per calcolare dt
    last_step = (step == max_steps - 1)
    
    #evalo la loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        print(f"val_loss: {val_loss_accum.item():.4f}")
        with open(log_file, 'a') as f:
            f.write(f"{step} val {val_loss_accum.item():.4f}\n")
        #salvo dei checkpoint ogni 1000 step
        if step > 0 and (step % 1000 == 0 or last_step):
            checkpoint_path = os.path.join(log_dir, f'checkpoint_{step:05d}.pt')
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'config': model.config,
                'val_loss': val_loss_accum.item()
            }
            torch.save(checkpoint, checkpoint_path)
            
    #train step
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    #normalizzo i gradienti se oltrepassano 1.0 per stabilizzare l'addestramento
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #determino il learning rate per l'iterazione corrente
    lr = get_lr(step)
    #aggiorno il learning rate dell'optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    #aspetto che la GPU finisca di calcolare i gradienti prima di fare l'optimizer step per colacare meglio il calcolo del tempo per iterazione
    if device_type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    #calcolo i token processati in questa iterazione (trainl_loader.B e train_loader.T sono gli stessi di B e T, ma li prendo da train_loader per leggibilità)
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    token_per_sec = tokens_processed / dt
    print(f'step {step}: loss = {loss_accum.item():.4f}, lr = {lr:.6f}, norm = {norm:.4f}, dt = {dt*1000:.2f}ms, tok/s = {token_per_sec:.2f}')
    with open(log_file, 'a') as f:
        f.write(f'{step} train {loss_accum.item():.4f}\n')
        