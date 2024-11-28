import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
from torch.utils.data import Subset
import os
import argparse 
import torchmetrics
import time
import json
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import EarlyStopping , create_experiment_folder , ContrastiveLoss
from configs import Configs 
from data import SiameseDataset
from models import  SiameseNetwork
from utils.optimizer import get_optimizer
from sklearn.model_selection import train_test_split

def train(cfg , model , criterion, optimizer, train_dataset, val_dataset):

    ##INitiate experiments
    
    exp_folder =  create_experiment_folder(base_path="experiments", config_data=cfg.__dict__)
    checkpoint_intervals = cfg.CHECKPOINT_INTERVAL

    early_stopping = EarlyStopping(patience=20 , min_delta=0.001)
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN_BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=cfg.VAL_BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
    # Initialize metrics
    # precision_metric = torchmetrics.Precision(num_classes= 1 , average="binary" , task="binary").to(cfg.DEVICE)
    # recall_metric = torchmetrics.Recall(num_classes= 1 , average="binary" , task="binary").to(cfg.DEVICE)
    mse = nn.MSELoss()
    
    start_time = time.time()

    for epoch in range(cfg.NUM_EPOCHS):
        model.train()
        train_loss = 0
        train_distance = 0
        train_corr = 0
        train_samples = 0
        model
        # precision_metric.reset()
        # recall_metric.reset()
        print("-------_TRAINING BEGIN------------")
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{cfg.NUM_EPOCHS}', unit='batch') as pbar:
            for imgs , labels in train_loader:
                img1 = imgs[0].to(cfg.DEVICE)
                img2 = imgs[1].to(cfg.DEVICE)
                labels = labels.to(cfg.DEVICE)
                # Forward pass
                optimizer.zero_grad()
                x1 , x2 = model(img1 , img2 )

                # Metric
                train_distance += mse(x1 , x2)


                # Calculate loss
                loss = criterion(x1 , x2 , labels)
                ## Update metrics
                train_loss += loss.item() * len(labels)
                train_samples += len(labels) 
                # Backward pass 9 Exploding Gradient)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(criterion.parameters()), 0.8)

                optimizer.step()

                pbar.set_postfix(loss=loss.item())  # Update the progress bar with loss
                pbar.update(1)  # Increment the progress bar by 1
        train_distance = train_distance / train_samples
        train_epoch_loss = train_loss / train_samples
        # train_precision = precision_metric.compute()
        # train_recall = recall_metric.compute()
        print(f"Epoch [{epoch + 1}/{cfg.NUM_EPOCHS}], Loss: {train_epoch_loss:.4f}, Distance: {train_distance:.4f} , ")
        print("-------_VALIDATION BEGIN------------")
        # Validate model
        model.eval()
        val_loss = 0
        val_corr = 0
        best_val_loss = 0
        total_samples = 0
        val_distance = 0

        # precision_metric.reset()
        # recall_metric.reset()

        with torch.no_grad():
            with tqdm(total=len(val_loader), desc='Validating', unit='batch') as pbar:
                for imgs, labels in val_loader:
                    img1 = imgs[0].to(cfg.DEVICE)
                    img2 = imgs[1].to(cfg.DEVICE)
                    labels = labels.to(cfg.DEVICE)
                    # Forward pass
                    optimizer.zero_grad()
                    x1 , x2 = model(img1 , img2 )

                    # Metric
                    val_distance += mse(x1 , x2)


                    # Calculate loss
                    loss = criterion(x1 , x2 , labels)
                    ## Update metrics
                    val_loss += loss.item() * len(labels)
                    total_samples += len(labels) 
                    # Backward pass 9 Exploding Gradient)

                    pbar.set_postfix(loss=loss.item())  # Update the progress bar with loss
                    pbar.update(1)  # Increment the progress bar by 1
        val_loss = val_loss / total_samples
        val_distance = val_distance / total_samples
        # val_precision = precision_metric.compute()
        # val_recall = recall_metric.compute()
        print(f"Epoch [{epoch + 1}/{cfg.NUM_EPOCHS}], Loss: {val_loss:.4f}, Distance: {val_distance:.4f} , ")
        
        if val_loss < best_val_loss:
            print("Saving best model")
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(exp_folder, "best", 'best_model.pth'))
            torch.save(criterion.state_dict(), os.path.join(exp_folder, 'best', 'best_criterion.pth'))
        if checkpoint_intervals > 0 and epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(exp_folder, "checkpoints", f"checkpoint_epoch_{epoch}.pt"))
            torch.save(criterion.state_dict(), os.path.join(exp_folder, 'checkpoints', f"checkpoint_epoch_{epoch}_criterion.pt"))

        # wandb.log({
        #     "train_loss": train_loss,
        #     "train_distance": train_distance,
        #     # "train_precision": train_precision,
        #     # "train_recall": train_recall,
        #     "val_loss": val_loss,
        #     "val_distance": val_distance,
        #     # "val_precision": val_precision,
        # })
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    #Save model
    torch.save(model.state_dict(), os.path.join(exp_folder,  "checkpoints", 'last_model.pt'))
    torch.save(criterion.state_dict(), os.path.join(exp_folder,  "checkpoints", 'last_criterion.pt'))
    # wandb.save(os.path.join(exp_folder, "checkpoints", 'last_model.pt'))
    # wandb.save(os.path.join(exp_folder, "checkpoints", 'last_criterion.pt'))
    # wandb.save(os.path.join(exp_folder, "best", 'best_model.pth'))
    # wandb.save(os.path.join(exp_folder, 'best', 'best_criterion.pth'))
    finish_time = time.time()
    total_training_time = finish_time - start_time
    print("Total training time: {:.2f} seconds".format(total_training_time))
    # wandb.log({"total_training_time": total_training_time})
    


def main():
    ## Initilization
    cfg = Configs()
    # Data Declaration

    transform = transforms.Compose([
        transforms.Resize((112,112)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
    ])


    # WandB
    # wandb.login(key=os.getenv("WANDB_API_KEY"))
    # wandb.init(project="face_recognition", entity="baihaqihilmi18-pamukkale-niversitesi")  # Replace with your WandB project and username

## Network Definition 

    network = SiameseNetwork()
    network.to(cfg.DEVICE)
    for param in network.parameters():
        param.requires_grad = True
        print(param.requires_grad)


## Dataset Init
    print("-----------DATASET LOADING--------")
    dataset = SiameseDataset(cfg.DATA_DIR, transform=transform)

    targets = dataset.labels
    train_idx, val_idx = train_test_split(range(len(targets)), test_size=0.2, stratify=targets)

    classes_ = dataset.class_to_idx()

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    # Classes JSON

    with open(os.path.join(cfg.DATA_DIR, 'classes.json'), 'w') as f:
        json.dump(classes_, f)

    print("-----------DATASET FINISHED--------")

    # Loss Function 

    criterion  = ContrastiveLoss()


    ## Optimizer Definition
    params = [{"params": network.parameters()}]
    optimizer = get_optimizer(cfg = cfg, params = params)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)

    # WandB
#     wandb.config.update({
#     "learning_rate": cfg.LEARNING_RATE,
#     "train_batch_size": cfg.TRAIN_BATCH_SIZE, 
#     "num_epochs": cfg.NUM_EPOCHS,
#     "Loss Function": cfg.LOSS,
#     "Model" : cfg.MODEL_NAME

# })
    
    
    # Train
    train(cfg , network, criterion, optimizer , train_dataset, val_dataset)
    # wandb.finish()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise e
    else:
        os.system("shutdown +2")
        pass