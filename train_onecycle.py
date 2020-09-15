import torch
from torch import optim
from torch import nn
from dataloader import get_imdb
from novograd import NovoGrad
import math
from model import Net
from lamb import Lamb
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

ap.add_argument("--num_heads", default=4, type=int, dest="num_heads", help="Number of attention heads in \
                                                                the Transformer network")

ap.add_argument("--num_blocks", default=1, type=int, dest="num_blocks",
                help="Number of blocks in the Transformer network")

ap.add_argument("--dropout", default=0.1, type=float, dest="dropout", help="Dropout (not keep_prob, but probability of ZEROING \
                                                                during training, i.e. keep_prob = 1 - dropout)")

ap.add_argument("--train_word_embeddings", type=bool, default=True, dest="train_word_embeddings",
                help="Train GloVE word embeddings")

ap.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
ap.add_argument('--weight-decay', type=float, default=5e-4, metavar='W',
                    help='SGD weight decay (default: 5e-4)')

ap.add_argument("--batch_size", type=int, default=128, help="Batch size")

ap.add_argument("--exp-rt",action='store_true', help="use exp-rt to see lr range")

ap.add_argument('--optimizer',type=str,default='sgd',
                    help='different optimizers')
ap.add_argument('--max-lr',default=1,type=float)
ap.add_argument('--pct-start',default=0.3,type=float)
ap.add_argument('--div-factor',default=25,type=float)
ap.add_argument('--final-div',default=10000,type=float)

ap.add_argument("--onecycle",action='store_true')

args = ap.parse_args()

# arg_pass = vars(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

loss_list = []
tacc = []
vacc = []

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
        vacc.append(correct/total)


def train(max_length, model_size,
          epochs, learning_rate,
          num_heads, num_blocks,
          dropout, train_word_embeddings,
          batch_size,exp_rt):
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

    # optimizer = NovoGrad((p for p in model.parameters() if p.requires_grad), lr=learning_rate,weight_decay=1e-4)
    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.optimizer.lower() == 'sgdwm':
        optimizer = optim.SGD((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate,
                                     weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'rmsprop':
        optimizer = optim.RMSprop((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, momentum=args.momentum,
                                  weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adagrad':
        optimizer = optim.Adagrad((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'lamb':
        from lamb import Lamb
        optimizer = Lamb((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'novograd':
        from novograd import NovoGrad
        optimizer = NovoGrad((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, momentum=args.momentum,
                              weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    def lrs(batch):
        low = math.log2(1e-5)
        high = math.log2(10)
        return 2 ** (low + (high - low) * batch / len(tqdm(train)) / args.epochs)

    if exp_rt:
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer,lrs)
    elif args.onecycle:
        lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer,args.learning_rate,steps_per_epoch=len(tqdm(train)),
                                                   epochs=args.epochs,div_factor=args.div_factor,final_div_factor=args.final_div,pct_start=args.pct_start)

    for i in range(epochs):
        loss_sum = 0.0
        correct = 0.0
        total = 0.0
        model.train()
        for j, b in enumerate(iter(tqdm(train))):
            print(j)
            optimizer.zero_grad()
            model_out = model(b.text[0].to(device))
            correct += (model_out.argmax(axis=1) == b.label.numpy()).sum()
            total += b.label.size(0)
            loss = criterion(model_out, b.label.to(device))
            loss.backward()
            optimizer.step()
            if exp_rt or args.onecycle:
                lr_scheduler.step()
                print('lr',lr_scheduler.get_last_lr())
                print('loss:',loss.item())
            loss_sum += loss.item()
            loss_list.append(loss.item())
        print("Epoch: {}, Loss mean: {}\n".format(i, j, loss_sum / j))
        tacc.append(correct/total)

        # Validate on test-set every epoch
        if not exp_rt:
            val(model, test, vocab, device)


if __name__ == "__main__":
    train(args.max_length,args.model_size,args.epochs,args.learning_rate,args.num_heads,args.num_blocks,args.dropout,args.train_word_embeddings,args.batch_size,args.exp_rt)

    postfix = '_exp-rt.json' if args.exp_rt else ('onecycle.json' if args.onecycle else '.json')
    if args.exp_rt:
        json.dump(loss_list,open('imdb_{}_batch{}_lr{}_epoch{}'.format(args.optimizer,args.batch_size,args.learning_rate,args.epochs)+ postfix,'w+'))
    else:
        json.dump([tacc,vacc],
                  open('imdb_{}_batch{}_lr{}_epoch{}'.format(args.optimizer,args.batch_size, args.learning_rate, args.epochs) + postfix,
                       'w+'))

