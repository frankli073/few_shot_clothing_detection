# File: train.py
# coding=utf-8
from prototypical_batch_sampler import PrototypicalBatchSampler
from prototypical_loss import PrototypicalLoss
from ClothingDataset import ClothingDataset
from protonet import ProtoNet
from parser_util import get_parser

from tqdm import tqdm
import numpy as np
import torch
import os


def init_seed(opt):
    """
    Disable cudnn for reproducibility.
    """
    torch.backends.cudnn.enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.manual_seed)


def init_dataloader(opt, mode):
    """
    Initialize DataLoader for training or validation.
    """
    if mode == 'train':
        dataset = ClothingDataset(mode=opt.train_dataset, root=opt.dataset_root)
        sampler = PrototypicalBatchSampler(
            labels=dataset.targets,
            classes_per_it=opt.classes_per_it_tr,
            num_samples=opt.num_support_tr + opt.num_query_tr,
            iterations=opt.iterations
        )
    elif mode == 'val':
        dataset = ClothingDataset(mode=opt.val_dataset, root=opt.dataset_root)
        sampler = PrototypicalBatchSampler(
            labels=dataset.targets,
            classes_per_it=opt.classes_per_it_val,
            num_samples=opt.num_support_val + opt.num_query_val,
            iterations=opt.iterations
        )
    else:
        raise ValueError("Mode must be 'train' or 'val'.")

    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    return dataloader


def init_protonet(opt):
    """
    Initialize the ProtoNet.
    """
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    model = ProtoNet(x_dim=3, hid_dim=64, z_dim=64).to(device)
    return model


def train(opt, tr_dataloader, model, optimizer, lr_scheduler, val_dataloader=None):
    """
    Train the Prototypical Network.
    """
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

    criterion = PrototypicalLoss(n_support=opt.num_support_tr).to(device)

    best_acc = 0
    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')

    for epoch in range(opt.epochs):
        print(f'=== Epoch: {epoch + 1}/{opt.epochs} ===')

        # Training phase
        model.train()
        train_loss, train_acc = [], []
        for batch in tqdm(tr_dataloader, desc="Training"):
            optimizer.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            embeddings = model(x)
            loss, acc = criterion(embeddings, y)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_acc.append(acc.item())

        avg_train_loss = np.mean(train_loss)
        avg_train_acc = np.mean(train_acc)
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")

        lr_scheduler.step()

        # Validation phase
        if val_dataloader:
            model.eval()
            val_loss, val_acc = [], []
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validation"):
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    embeddings = model(x)
                    loss, acc = criterion(embeddings, y)

                    val_loss.append(loss.item())
                    val_acc.append(acc.item())

            avg_val_loss = np.mean(val_loss)
            avg_val_acc = np.mean(val_acc)
            print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

            if avg_val_acc > best_acc:
                best_acc = avg_val_acc
                torch.save(model.state_dict(), best_model_path)

    torch.save(model.state_dict(), last_model_path)
    print(f"Training complete. Best Val Acc: {best_acc:.4f}")


def main():
    """
    Main training function.
    """
    options = get_parser().parse_args()
    os.makedirs(options.experiment_root, exist_ok=True)

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: CUDA device is available but not enabled. Run with --cuda to enable.")

    init_seed(options)

    # Use `top_10` for initial training and validation
    options.train_dataset = 'top_10'
    options.val_dataset = 'top_10'

    tr_dataloader = init_dataloader(options, mode='train')
    val_dataloader = init_dataloader(options, mode='val')

    model = init_protonet(options)
    optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=options.lr_scheduler_step, gamma=options.lr_scheduler_gamma)

    train(
        opt=options,
        tr_dataloader=tr_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )


if __name__ == '__main__':
    main()
