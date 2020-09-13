import torch
from torch import optim
from torch import nn
from dataloader import get_imdb
from novograd import NovoGrad
import math
from model import Net
import argparse
import json

ap = argparse.ArgumentParser(description="Train a Transformer network for sentiment analysis")

ap.add_argument("--max_length", default=500, type=int, help="Maximum sequence length, \
                                                                sequences longer than this are truncated")

ap.add_argument("--model_size", default=128, type=int, help="Hidden size for all \
                                                                hidden layers of the model")

ap.add_argument("--epochs", default=1000, type=int, help="Number of epochs to train for")

ap.add_argument("--learning_rate", default=0.001, type=float, dest="learning_rate",
                help="Learning rate for optimizer")

ap.add_argument("--device", default="cuda:0", dest="device", help="Device to use for training \
                                                                and evaluation e.g. (cpu, cuda:0)")
ap.add_argument("--num_heads", default=4, type=int, dest="num_heads", help="Number of attention heads in \
                                                                the Transformer network")

ap.add_argument("--num_blocks", default=1, type=int, dest="num_blocks",
                help="Number of blocks in the Transformer network")

ap.add_argument("--dropout", default=0.1, type=float, dest="dropout", help="Dropout (not keep_prob, but probability of ZEROING \
                                                                during training, i.e. keep_prob = 1 - dropout)")

ap.add_argument("--train_word_embeddings", type=bool, default=True, dest="train_word_embeddings",
                help="Train GloVE word embeddings")

ap.add_argument("--batch_size", type=int, default=128, help="Batch size")

ap.add_argument("--exp-rt",action='store_ture', help="use exp-rt to see lr range")

ap.add_argument("--onecycle",action='store_true')

args = ap.parse_args()

arg_pass = vars(args)

loss_list = []


try:
    # try to import tqdm for progress updates
    from tqdm import tqdm
except ImportError:
    # on failure, make tqdm a noop
    def tqdm(x):
        return x

try:
    # try to import visdom for visualisation of attention weights
    import visdom
    from helpers import plot_weights

    vis = visdom.Visdom()
except ImportError:
    vis = None
    pass


def val(model, test, vocab, device):
    """
        Evaluates model on the test set
    """
    model.eval()

    print("\nValidating..")

    if not vis is None:
        visdom_windows = None
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        for i, b in enumerate(tqdm(test)):
            if not vis is None and i == 0:
                visdom_windows = plot_weights(model, visdom_windows, b, vocab, vis)

            model_out = model(b.text[0].to(device)).to("cpu").numpy()
            correct += (model_out.argmax(axis=1) == b.label.numpy()).sum()
            total += b.label.size(0)
        print("{}%, {}/{}".format(correct / total, correct, total))

def lrs(batch):
    low = math.log2(1e-5)
    high = math.log2(10)
    return 2 ** (low + (high - low) * batch / len(tqdm(train)) / args.epochs)

def train(max_length, model_size,
          epochs, learning_rate,
          device, num_heads, num_blocks,
          dropout, train_word_embeddings,
          batch_size,exp_rt,onecycle):
    """
        Trains the classifier on the IMDB sentiment dataset
    """
    train, test, vectors, vocab = get_imdb(batch_size, max_length=max_length)

    model = Net(
        model_size=model_size, embeddings=vectors,
        max_length=max_length, num_heads=num_heads,
        num_blocks=num_blocks, dropout=dropout,
        train_word_embeddings=train_word_embeddings,
    ).to(device)

    optimizer = NovoGrad((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    if exp_rt:
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer,lrs)
    elif args.onecycle:
        lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer,args.learning_rate,steps_per_epoch=len(tqdm(train)),
                                                   epochs=args.epochs)

    for i in range(0, epochs + 1):
        loss_sum = 0.0
        model.train()
        for j, b in enumerate(iter(tqdm(train))):
            optimizer.zero_grad()
            model_out = model(b.text[0].to(device))
            loss = criterion(model_out, b.label.to(device))
            loss.backward()
            optimizer.step()
            if exp_rt or args.onecycle:
                lr_scheduler.step()
            loss_sum += loss.item()
            loss_list.append(loss.item())
        print("Epoch: {}, Loss mean: {}\n".format(i, j, loss_sum / j))

        # Validate on test-set every epoch
        if not exp_rt:
            val(model, test, vocab, device)


if __name__ == "__main__":
    train(**arg_pass)

    postfix = '_exp-rt.json' if args.exp_rt else ('onecycle.json' if args.onecycle else '.json')
    json.dump(loss_list,open('imdb_batch{}_lr{}_epoch{}'.format(args.batch_size,args.learning_rate,args.epochs)+ postfix),'w+')


