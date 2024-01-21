import time

out_dir = 'out-apachelogs'
eval_interval = 200
eval_iters = 1
# init_from = 'gpt2-xl' # this is the largest GPT-2 model

wandb_project = 'apachelogs'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'apachelogs'

batch_size=8
block_size=64

n_layer=4
n_head=4
n_embd=64

# learning_rate = 3e-5

device = 'mps'
compile=False