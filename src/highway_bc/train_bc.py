import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from highway_bc.dataset import ExpertDataset
from highway_bc.policy import BCPolicy

# global seed: controls weight initialization and DataLoader shuffling
torch.manual_seed(42)

# hyperparameters --> control of model optimization process
EPOCHS = 60             # number of iterations through the whole dataset
BATCH_SIZE = 64         # number of examples before network parameters are updated
LEARNING_RATE = 1e-3    # how much to update model parameters each batch --> carefull

def train_loop(dataloader, model, loss_fn, optimizer):
    """
    Trains the model for one epoch over the whole training set.

    Args:
        dataloader: yields batches of (X, y); X = [B, 25] observations, y = [B] expert actions
        model: BCPolicy to train
        loss_fn: CrossEntropyLoss
        optimizer: Adam, updates the model parameters

    Returns:
        float: average loss per batch over this epoch
    """
    size = len(dataloader.dataset)      # number of samples
    model.train()
    num_batches = len(dataloader)       # number of batches, used for the average below
    total_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        # compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation --> actual training
        optimizer.zero_grad()   # clear gradients of the previous batch (PyTorch accumulates them)
        loss.backward()         # compute how much each parameter contributed to the loss
        optimizer.step()        # move each parameter a small step against its gradient

        # collect the per-batch losses to average them at the end of the epoch
        total_loss += loss.item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * BATCH_SIZE + len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    return total_loss / num_batches

def v_loop(dataloader, model, loss_fn):
    """
    Evaluates the model with the validation set. No parameter updates.

    Also prints a per-class table, because overall accuracy is misleading on this
    dataset: ~70% of all expert actions are IDLE, so always predicting IDLE already
    scores 70%.

    Args:
        dataloader: yields batches of (X, y) from the validation split
        model: BCPolicy to evaluate
        loss_fn: CrossEntropyLoss

    Returns:
        tuple: (val_loss, accuracy)
            val_loss: average loss per batch
            accuracy: fraction of correctly predicted actions, 0.0 to 1.0
    """
    size = len(dataloader.dataset)
    model.eval()
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # per-class counters, one entry per discrete action
    n_classes = 5
    pred_counts = torch.zeros(n_classes)    # how often the model predicts each action
    true_counts = torch.zeros(n_classes)    # how often each action actually occurs
    hit_counts = torch.zeros(n_classes)     # how often the model gets each action right

    # evaluation model with torch.no_grad --> no gradient computation
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # argmax(1) = predicted action per row; compare with y and count the hits
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            predicted = pred.argmax(1)
            pred_counts += torch.bincount(predicted, minlength=n_classes)
            true_counts += torch.bincount(y, minlength=n_classes)
            # y[predicted == y] keeps only the labels that were predicted correctly
            hit_counts += torch.bincount(y[predicted == y], minlength=n_classes)            

    test_loss /= num_batches    # losses were summed per batch
    correct /=size              # hits were counted per sample

    action_names = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]
    print(f"{'action':<12}{'in data':>9}{'predicted':>11}{'recall':>9}")
    for i, name in enumerate(action_names):
        # recall = share of actual occurrences that the model found
        recall = hit_counts[i] / true_counts[i] if true_counts[i] > 0 else 0.0
        print(f"{name:<12}{int(true_counts[i]):>9}{int(pred_counts[i]):>11}{100*recall:>8.1f}%")
    
    return test_loss, correct

def main():
    """Loads the expert demonstrations, trains the BC policy and saves the weights."""
    dataset = ExpertDataset("data/demos_1000.npz")

    # own generator so the split stays identical no matter what else draws random numbers
    generator = torch.Generator().manual_seed(42)
    # Split the dataset into two subsets: 80% training and 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator)

    # shuffle only for training, so the model does not learn the sample order
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = BCPolicy()

    # outer epoche loop
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n----------------")
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        val_loss, val_acc = v_loop(val_dataloader, model, loss_fn)
        print(f"train loss: {train_loss:>7f} | val loss: {val_loss:>7f} | val acc: {(100*val_acc):>0.1f}%\n")
    print("Done!")

    # TODO: save the best epoch instead of the last one; this overwrites the file on every run
    torch.save(model.state_dict(), 'checkpoints/bc_policy.pth')

if __name__ == "__main__":
    main()