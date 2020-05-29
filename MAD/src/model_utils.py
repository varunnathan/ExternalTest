import time
import torch
import torch.optim as torch_optim
import torch.nn.functional as F
from torch import tensor
from sklearn.metrics import roc_auc_score


def choose_embedding_size(cat_cols, cat_num_values, min_emb_dim=100):
    """
    cat_cols: list of categorical columns
    cat_num_values: list of number of unique values for each categorical column
    """

    embedded_cols = dict(zip(cat_cols, cat_num_values))
    embedding_sizes = [(n_categories, min(min_emb_dim, (n_categories+1)//2))
                       for _, n_categories in embedded_cols.items()]
    return embedding_sizes


def get_default_device():
    """Pick GPU if available, else CPU"""

    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""

    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def get_optimizer(model, lr = 0.001, wd = 0.0):

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optim = torch_optim.Adam(parameters, lr=lr, weight_decay=wd)
    return optim


def construct_tensor(a):

     final = []
     for i in a:
         out = []
         for j in i:
             out.append(j.tolist())
         out1 = []
         for item in zip(*out):
             out1.append(list(item))
         final += out1
     return tensor(final)


def construct_tensor_y(a):

     out = []
     for i in a:
         out += i.tolist()
     return tensor(out)


def train_model(model, optim, train_dl, train_size, chunksize, batch_size,
                device, loss_fn=F.cross_entropy):

    model.train()
    total = 0
    sum_loss = 0
    with tqdm(total=train_size // (batch_size * chunksize)) as pbar:
        for x1, x2, y in train_dl:
            x1, x2, y = (construct_tensor(x1), construct_tensor(x2),
                         construct_tensor_y(y))
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            batch = y.size()[0]
            output = model(x1, x2)
            loss = loss_fn(output, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += batch
            sum_loss += loss.item()
            pbar.update(1)
    return sum_loss/total


def val_loss(model, valid_dl, test_size, chunksize, batch_size,
             device, loss_fn=F.cross_entropy):

    model.eval()
    total = 0
    sum_loss = 0
    correct = 0
    sum_auc_macro = 0
    sum_auc_micro = 0
    num_aucs = 0
    with tqdm(total=test_size // (batch_size * chunksize)) as pbar:
        for x1, x2, y in valid_dl:
            x1, x2, y = (construct_tensor(x1), construct_tensor(x2),
                         construct_tensor_y(y))
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            batch = y.size()[0]
            out = model(x1, x2)
            loss = loss_fn(out, y)
            sum_loss += loss.item()
            total += batch
            pred = torch.max(out, 1)[1]
            pred_prob = F.softmax(out, dim=1)
            y_onehot = F.one_hot(y)
            correct += (pred == y).float().sum().item()
            pred_prob = pred_prob.cpu().detach().numpy()
            y_onehot = y_onehot.cpu().detach().numpy()
            try:
                sum_auc_macro += roc_auc_score(y_onehot, pred_prob,
                                               average='macro')
                sum_auc_micro += roc_auc_score(y_onehot, pred_prob,
                                               average='micro')
                num_aucs += 1
            except:
                continue

            pbar.update(1)
    print("valid loss %.3f, accuracy %.3f, macro auc %.3f and micro auc %.3f" % (
        sum_loss/total, correct/total, sum_auc_macro/num_aucs, sum_auc_micro/num_aucs))
    return sum_loss/total, correct/total, sum_auc_macro/num_aucs, sum_auc_micro/num_aucs


def train_loop(model, train_dl, valid_dl, epochs, train_size,
               test_size, chunksize, batch_size, device, lr=0.01,
               wd=0.0, loss_fn=F.cross_entropy):

    optim = get_optimizer(model, lr = lr, wd = wd)
    start = time.time()
    losses = []
    for i in range(epochs):
        stats = {'epoch': i+1}
        train_loss = train_model(model, optim, train_dl, train_size,
                                 chunksize, batch_size, device,
                                 loss_fn)
        print("training loss: ", train_loss)
        stats['train_loss'] = train_loss
        loss, acc, auc_macro, auc_micro = val_loss(
            model, valid_dl, test_size, chunksize, batch_size, device, loss_fn)
        print('time taken: %0.2f' % (time.time() - start))
        stats['test_loss'] = loss
        stats['test_acc'] = acc
        stats['test_auc_macro'] = auc_macro
        stats['test_auc_micro'] = auc_micro
        losses.append(stats)
    return losses
