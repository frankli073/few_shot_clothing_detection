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
    Initialize DataLoader for training, validation, or testing.
    """
    dataset = ClothingDataset(mode=opt.train_dataset if mode == 'train' else opt.val_dataset, root=opt.dataset_root)
    sampler = PrototypicalBatchSampler(
        labels=dataset.targets,
        classes_per_it=opt.classes_per_it_tr if mode == 'train' else opt.classes_per_it_val,
        num_samples=(opt.num_support_tr + opt.num_query_tr) if mode == 'train' else (opt.num_support_val + opt.num_query_val),
        iterations=opt.iterations
    )
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
    Train the Prototypical Network and log results.
    """
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    criterion = PrototypicalLoss(n_support=opt.num_support_tr).to(device)

    # Initialize logging variables
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    # Create dynamic output folder
    mode_folder = f"{opt.train_dataset}_train_{opt.val_dataset}_val"
    output_folder = os.path.join(
        "C:\\work\\few_shot_clothing_detection\\models\\prototypical\\output", mode_folder
    )
    os.makedirs(output_folder, exist_ok=True)

    # Define model checkpoint filenames
    best_model_path = os.path.join(output_folder, 'best_model.pth')
    last_model_path = os.path.join(output_folder, 'last_model.pth')

    best_acc = 0

    for epoch in range(opt.epochs):
        print(f'=== Epoch: {epoch + 1}/{opt.epochs} ===')

        # Training Phase
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
        train_loss_history.append(avg_train_loss)
        train_acc_history.append(avg_train_acc)
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")

        lr_scheduler.step()

        # Validation Phase
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
            val_loss_history.append(avg_val_loss)
            val_acc_history.append(avg_val_acc)
            print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

            # Save the best model
            if avg_val_acc > best_acc:
                best_acc = avg_val_acc
                torch.save(model.state_dict(), best_model_path)

    # Save the final model
    torch.save(model.state_dict(), last_model_path)

    # Save metrics
    np.save(os.path.join(output_folder, 'train_loss.npy'), train_loss_history)
    np.save(os.path.join(output_folder, 'train_acc.npy'), train_acc_history)
    np.save(os.path.join(output_folder, 'val_loss.npy'), val_loss_history)
    np.save(os.path.join(output_folder, 'val_acc.npy'), val_acc_history)

    print(f"Training complete. Best Val Acc: {best_acc:.4f}")


def main():
    """
    Main training function.
    """
    options = get_parser().parse_args()

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: CUDA device is available but not enabled. Run with --cuda to enable.")

    init_seed(options)

    # Use dataset modes for dynamic folder naming
    options.train_dataset = 'top_10'  # Change this as needed
    options.val_dataset = 'top_10'    # Change this as needed

    tr_dataloader = init_dataloader(options, mode='train')
    val_dataloader = init_dataloader(options, mode='val')

    model = init_protonet(options)
    optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate, weight_decay=1e-4)  # Added weight decay
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
