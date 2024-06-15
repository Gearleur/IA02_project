import os
import torch

def save_checkpoint(model, optimizer, iteration, save_dir='models'):
    # Créer le répertoire s'il n'existe pas
    os.makedirs(save_dir, exist_ok=True)
    
    # Chemins des fichiers à sauvegarder
    model_path = os.path.join(save_dir, f'model_gopher_parallel_{iteration}.pt')
    optimizer_path = os.path.join(save_dir, f'optimizer_gopher_parallel_{iteration}.pt')

    # Sauvegarde des états du modèle et de l'optimizer
    torch.save(model.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optimizer_path)
    print(f"Model and optimizer saved to {model_path} and {optimizer_path}")

def load_checkpoint(model, optimizer, iteration, load_dir='models'):
    model_path = os.path.join(load_dir, f'model_gopher_parallel_{iteration}.pt')
    optimizer_path = os.path.join(load_dir, f'optimizer_gopher_parallel_{iteration}.pt')

    model.load_state_dict(torch.load(model_path))
    optimizer.load_state_dict(torch.load(optimizer_path))
    print(f"Model and optimizer loaded from {model_path} and {optimizer_path}")