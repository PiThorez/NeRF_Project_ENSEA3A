import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F

# Hyperparamètres
num_layers = 8  # Nombre de couches dans le MLP
hidden_size = 256  # Taille des couches cachées
learning_rate = 5e-4  # Taux d'apprentissage pour l'optimisation
num_epochs = 10  # Nombre d'époques pour l'entraînement
batch_size = 1024  # Taille du lot

# Fonction pour charger les images depuis le dossier
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img = Image.open(img_path).convert('RGB')  # Assurez-vous que l'image est en RGB
            images.append(np.array(img))
    return images

# Fonction pour charger les positions des caméras depuis un fichier texte
def load_camera_positions(file_path):
    positions = []
    with open(file_path, 'r') as file:
        for line in file:
            data = list(map(float, line.strip().split()))
            positions.append(data)
    return positions

# Classe du modèle NeRF
class NeRF(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(NeRF, self).__init__()
        self.input_dim = 3 + 3  # Position (x, y, z) et direction (dx, dy, dz)
        layers = []
        
        # Couche d'entrée
        layers.append(nn.Linear(self.input_dim, hidden_size))
        layers.append(nn.ReLU())
        
        # Couches cachées
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        # Couche de sortie
        layers.append(nn.Linear(hidden_size, 4))  # RGB et densité (sigma)
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Fonction pour générer des rayons 3D à partir des positions de la caméra et des images 2D
def generate_rays_from_images_and_cameras(images, camera_positions):
    rays = []
    for img, cam_pos in zip(images, camera_positions):
        # Exemple simplifié pour générer un rayon
        rays_for_image = []
        height, width, _ = img.shape
        for i in range(height):
            for j in range(width):
                # Chaque pixel de l'image génère un rayon
                direction = np.array([0, 0, 1])  # Rayon dans une direction fixe (simplification)
                ray_origin = np.array(cam_pos[:3])  # Position de la caméra
                rays_for_image.append((ray_origin, direction))
        rays.append(rays_for_image)
    return rays  # Retourner directement les rayons (une liste de listes)

# Fonction de rendu 3D (simplification)
def render_3d_scene(rays, nerf_model, device):
    nerf_model.eval()
    all_rendered_images = []
    for ray_set in rays:
        rendered_image = []
        for ray_origin, ray_direction in ray_set:
            ray_origin = torch.tensor(ray_origin).to(device).float()  # Assurez-vous que c'est float32
            ray_direction = torch.tensor(ray_direction).to(device).float()  # Assurez-vous que c'est float32
            ray_input = torch.cat([ray_origin, ray_direction], dim=-1).unsqueeze(0)  # Ajouter une dimension pour le batch
            color_and_density = nerf_model(ray_input)
            rendered_image.append(color_and_density.cpu().detach())
        all_rendered_images.append(torch.cat(rendered_image, dim=0))  # Concaténer les résultats pour chaque image
    return all_rendered_images

# Fonction d'entraînement du modèle NeRF
def train_nerf(device):
    # Initialisation du modèle NeRF et de l'optimiseur
    nerf_model = NeRF(hidden_size, num_layers).to(device)
    optimizer = torch.optim.Adam(nerf_model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        nerf_model.train()
        optimizer.zero_grad()

        # Générer des rayons à partir des images et des caméras
        rays = generate_rays_from_images_and_cameras(images, camera_positions)
        
        # Effectuer une passe avant pour obtenir les couleurs et densités
        color_and_density = render_3d_scene(rays, nerf_model, device)

        # Assurez-vous que color_and_density est un tensor et que sa forme est correcte
        color_and_density = torch.cat(color_and_density, dim=0)  # Concaténer les résultats des différentes images

        # Simuler une vérité de terrain (valeurs RGB et densité aléatoires ici)
        # Utilisez la taille de color_and_density pour générer un ground_truth de la même forme
        ground_truth = torch.rand(color_and_density.size(0), 4).to(device)  # Taille correcte

        # Vérifier si les tensors nécessitent des gradients
        if not color_and_density.requires_grad:
            color_and_density.requires_grad_()  # Permettre à ce tensor de suivre les gradients
        if not ground_truth.requires_grad:
            ground_truth.requires_grad_()  # Permettre à ce tensor de suivre les gradients

        # Calculer la perte (MSE pour RGB et densité)
        loss = F.mse_loss(color_and_density, ground_truth)

        # Rétropropagation et optimisation
        loss.backward()
        optimizer.step()

        if epoch % 1 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

    return nerf_model




# Exécution principale
if __name__ == "__main__":
    # Dossier contenant les images et fichier des caméras
    images_folder = 'images'
    camera_file_path = 'camera_positions.txt'  # Fichier texte avec les positions des caméras
    
    # Charger les images et les positions des caméras
    images = load_images_from_folder(images_folder)
    camera_positions = load_camera_positions(camera_file_path)
    
    # Sélectionner le périphérique (GPU si disponible, sinon CPU)
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")  # Devrait afficher 'cuda' si le GPU est utilisé

    
    # Entraîner le modèle NeRF
    trained_nerf = train_nerf(device)
    
    # Générer les rayons à partir des images et des caméras
    rays = generate_rays_from_images_and_cameras(images, camera_positions)
    
    # Créer un rendu 3D
    rendered_images = render_3d_scene(rays, trained_nerf, device)
    
    # Afficher un rendu pour vérification
    # Sélectionner la première image du rendu 3D et afficher les RGB (en prenant juste les 3 premières valeurs)
    plt.imshow(rendered_images[0].cpu().detach().numpy()[:, :3])  # Afficher la première image RGB
    plt.show()
