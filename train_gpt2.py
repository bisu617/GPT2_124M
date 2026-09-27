import os
from dataclasses import dataclass
import inspect
import math
from time import time
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads
        self.c_attn = nn.Linear(
            config.n_embd,
            3 * config.n_embd
        )

        # output projection
        self.c_proj = nn.Linear(
            config.n_embd,
            config.n_embd
        )
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # causal mask
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones(
                    config.block_size,
                    config.block_size
                )
            ).view(
                1,
                1,
                config.block_size,
                config.block_size
            )
        )

    def forward(self, x):

        B, T, C = x.size()

        # calculate query, key, value
        qkv = self.c_attn(x)

        q, k, v = qkv.split(
            self.n_embd,
            dim=2
        )

        k = k.view(
            B,
            T,
            self.n_head,
            C // self.n_head
        ).transpose(1, 2)

        q = q.view(
            B,
            T,
            self.n_head,
            C // self.n_head
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.n_head,
            C // self.n_head
        ).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = (
            y.transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        y = self.c_proj(y)

        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.c_fc = nn.Linear(
            config.n_embd,
            4 * config.n_embd
        )

        self.gelu = nn.GELU(
            approximate="tanh"
        )

        self.c_proj = nn.Linear(
            4 * config.n_embd,
            config.n_embd
        )
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):

        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(
            config.n_embd
        )

        self.attn = CausalSelfAttention(
            config
        )

        self.ln_2 = nn.LayerNorm(
            config.n_embd
        )

        self.mlp = MLP(config)

    def forward(self, x):

        x = x + self.attn(
            self.ln_1(x)
        )

        x = x + self.mlp(
            self.ln_2(x)
        )

        return x


@dataclass
class GPTConfig:

    block_size: int = 1024
    vocab_size: int = 50257

    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(dict(

            wte=nn.Embedding(
                config.vocab_size,
                config.n_embd
            ),

            wpe=nn.Embedding(
                config.block_size,
                config.n_embd
            ),

            h=nn.ModuleList([
                Block(config)
                for _ in range(config.n_layer)
            ]),

            ln_f=nn.LayerNorm(
                config.n_embd
            ),
        ))

        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False
        )

        #weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight   

        #init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std  = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):

        assert model_type in {
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl"
        }

        from transformers import GPT2LMHeadModel

        print(
            "loading weights from pretrained GPT:",
            model_type
        )

        config_args = {

            "gpt2": dict(
                n_layer=12,
                n_head=12,
                n_embd=768
            ),

            "gpt2-medium": dict(
                n_layer=24,
                n_head=16,
                n_embd=1024
            ),

            "gpt2-large": dict(
                n_layer=36,
                n_head=20,
                n_embd=1280
            ),

            "gpt2-xl": dict(
                n_layer=48,
                n_head=25,
                n_embd=1600
            ),

        }[model_type]

        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024

        config = GPTConfig(**config_args)

        model = GPT(config)

        sd = model.state_dict()

        sd_keys = [
            k
            for k in sd.keys()
            if not k.endswith("attn.bias")
        ]

        # HuggingFace GPT-2
        model_hf = GPT2LMHeadModel.from_pretrained(
            model_type
        )

        sd_hf = model_hf.state_dict()

        sd_keys_hf = [
            k
            for k in sd_hf.keys()
            if not k.endswith("attn.bias")
            and not k.endswith("attn.masked_bias")
        ]

        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight"
        ]

        assert len(sd_keys_hf) == len(sd_keys), (
            f"mismatch keys: "
            f"{len(sd_keys_hf)} vs "
            f"{len(sd_keys)}"
        )

        for k in sd_keys:

            if any(
                k.endswith(w)
                for w in transposed
            ):

                assert (
                    sd_hf[k].shape[::-1]
                    == sd[k].shape
                )

                with torch.no_grad():
                    sd[k].copy_(
                        sd_hf[k].t()
                    )

            else:

                assert (
                    sd_hf[k].shape
                    == sd[k].shape
                )

                with torch.no_grad():
                    sd[k].copy_(
                        sd_hf[k]
                    )

        return model

def configure_optimizer(self, weight_decay, learning_rate, device):
    
    param_dict = {pn: p for pn, p in self.named_parameters()}
    param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items()if p.dim()>=2]
    nodecay_params = [p for n, p in param_dict.items()if p.dim()<2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameters: {len(decay_params)}, with {num_decay_params:,} elements")
    print(f"num non-decayed parameters: {len(nodecay_params)}, with {num_nodecay_params:,} elements")
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    used_fused = fused_available and device == 'cuda'
    print(f"using fused AdamW: {used_fused}")
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=used_fused)
    return optimizer

GPT.configure_optimizers = configure_optimizer

import tiktoken
import numpy as np


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        data_root = "edu_fineweb_small"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")

        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def get_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# run the training loop
from torch.distributed import init_process_group, destroy_process_group

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?

if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"

    init_process_group(backend='nccl')

    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])

    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)

    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.

else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    # attempt to autodetect device
    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    print(f"using device: {device}")

torch.manual_seed(1337)

if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

import time

total_batch_size = 32768
B = 4
T = 1024
assert total_batch_size % (B * T) == 0, "total_batch_size must be divisible by (B * T)"
grad_accum_steps = total_batch_size // (B * T)
if master_process:
    print(f"total desired batch size: {total_batch_size:,}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

print("I am GPU", ddp_rank)

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")
torch.set_float32_matmul_precision("high")

enc = tiktoken.get_encoding("gpt2")

# ---- logging setup for the loss chart ----
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
with open(log_file, "w") as f:  # clear/create the file
    pass

model = GPT(GPTConfig(vocab_size = 50304))
model.to(device="cuda")
import time

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 200
def get_lr(it):
    if it<warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it>max_steps:
        return min_lr
    decay_ratio = (it-warmup_steps) / (max_steps-warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 +math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

#optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95),eps=1e-8)
optimizer = model.configure_optimizers(weight_decay = 0.1, learning_rate = 6e-4, device = torch.device("cuda"))

val_every = 20  # how often (in steps) to compute a validation loss
val_loss_steps = 5  # number of val batches to average each time

for step in range(max_steps):
    t0 = time.time()

    # periodic validation loss, for the chart
    if step % val_every == 0 or step == max_steps - 1:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            for _ in range(val_loss_steps):
                x, y = val_loader.get_batch()
                x, y = x.to("cuda"), y.to("cuda")
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.item()
        print(f"step {step}: val loss {val_loss_accum:.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} val {val_loss_accum:.6f}\n")
        model.train()

    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.get_batch()
        x, y = x.to("cuda"), y.to("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.item()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()  # wait for GPU to finish work before measuring time
    t1 = time.time()
    dt = (t1 - t0) * 1000
    print(f"step {step}: loss {loss_accum:.6f}, norm:{norm:.4f} time: {dt:.2f}ms")
    with open(log_file, "a") as f:
        f.write(f"{step} train {loss_accum:.6f}\n")

# save a checkpoint so we don't have to retrain from scratch next time
checkpoint_path = os.path.join(log_dir, "model_last.pt")
torch.save({
    "model": model.state_dict(),
    "config": model.config,
    "step": max_steps,
}, checkpoint_path)
print(f"saved checkpoint to {checkpoint_path}")

model.eval()
num_return_sequences = 5
max_length = 30
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to("cuda")

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits, loss = model(x)  # B, T, vocab_size
        logits = logits[:, -1, :]  # B, vocab_size
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

        for i in range(num_return_sequences):
            tokens = x[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(">", decoded)