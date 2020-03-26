import os, torch

from model import StyleEstimator
from test_tube import HyperOptArgumentParser

parser = HyperOptArgumentParser(
        strategy="random_search",
        description="Minimalist BERT Classifier",
        add_help=False,
    )
parser.add_argument("--seed", type=int, default=3, help="Training seed.")
parser.add_argument(
    "--save_top_k",
    default=1,
    type=int,
    help="The best k models according to the quantity monitored will be saved.",
)
# Early Stopping
parser.add_argument(
    "--monitor", default="val_loss", type=str, help="Quantity to monitor."
)
parser.add_argument(
    "--metric_mode",
    default="min",
    type=str,
    help="If we want to min/max the monitored quantity.",
    choices=["auto", "min", "max"],
)
parser.add_argument(
    "--patience",
    default=3,
    type=int,
    help="Number of epochs with no improvement \
        after which training will be stopped.",
)
parser.add_argument(
    "--min_epochs",
    default=1,
    type=int,
    help="Limits training to a minimum number of epochs",
)
parser.add_argument(
    "--max_epochs",
    default=20,
    type=int,
    help="Limits training to a max number number of epochs",
)

# Batching
parser.add_argument(
    "--batch_size", default=8, type=int, help="Batch size to be used."
)
parser.add_argument(
    "--accumulate_grad_batches",
    default=1,
    type=int,
    help="Accumulated gradients runs K small batches of size N before \
        doing a backwards pass.",
)

# gpu args
parser.add_argument("--gpus", type=int, default=0, help="How many gpus")
parser.add_argument(
    "--distributed_backend",
    type=str,
    default="dp",
    help="Supports three options dp, ddp, ddp2",
)
parser.add_argument(
    "--use_16bit",
    dest="use_16bit",
    action="store_true",
    help="If true uses 16 bit precision",
)
parser.add_argument(
    "--log_gpu_memory",
    type=str,
    default=None,
    help="Uses the output of nvidia-smi to log GPU usage. \
        Might slow performance.",
)

parser.add_argument(
    "--val_percent_check",
    default=1.0,
    type=float,
    help="If you don't want to use the entire dev set (for debugging or \
        if it's huge), set how much of the dev set you want to use with this flag.",
)


PATH = "/Users/sdumitre/work/NLP-Cube/style/experiments/lightning_logs/version_25-03-2020--15-56-50/_ckpt_epoch_0.ckpt"
tags_csv = "/Users/sdumitre/work/NLP-Cube/style/experiments/lightning_logs/version_25-03-2020--15-56-50/meta_tags.csv"
model = StyleEstimator.load_from_checkpoint(PATH, tags_csv = tags_csv)
model.eval()

x_tensor = torch.load("x.x")
x_lengths = torch.load("x.len")
print("we have {} examples".format(x_tensor.size(0)))
style = model(x_tensor, x_lengths)
print(">>>")
print(style)