import os
import nibabel as nib
from nibabel.orientations import io_orientation, axcodes2ornt, ornt2axcodes
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision import transforms
import numpy as np
from sklearn.model_selection import StratifiedKFold
from resnet import resnet18  # Assicurati che il tuo script resnet.py contenga questa funzione
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchsummary import summary
import random
import torchio as tio
from contextlib import redirect_stdout


# Percorso al file Excel contenente le eticchette
excel_path = r'C:\Users\marti\Desktop\besta\TLE\TLE_49_56\R1\dati_cnr_new.xlsx'
df = pd.read_excel(excel_path, usecols=['ID_CLOUD', 'TLE'])

# Mappa filename → label numerico (0 → 0, 1 → 1)
label_mapping = {row['ID_CLOUD']: 1 if row['TLE'] == 1 else 0 for _, row in df.iterrows()}

# Parametri di training
num_epochs = 50
batch_size = 8
lr = 0.001
patience = 10
patience_lr = 5
num_folds = 5

# Directory dei dati
root_dir = r"C:\Users\marti\Desktop\besta\TLE\TLE_49_56\R1_not_registered"

# Caricamento dati e label
all_files = [file for file in os.listdir(root_dir) if file.endswith(".nii")]
all_labels = [label_mapping[os.path.splitext(file)[0]] for file in all_files]

# >>> QUI: stampa dimensioni originali
print("Dimensioni originali dei volumi:")
print("File trovati:", all_files)

for file in all_files:
    img_path = os.path.join(root_dir, file)
    img_nib = nib.load(img_path)
    data = img_nib.get_fdata()
    file_id = os.path.splitext(file)[0]
    label = label_mapping.get(file_id, "Label non trovata")
    print(f"{file}: {data.shape}, Label: {label}")

# Trasformazione di preprocessing unica: z-score normalization
preprocess_transform = tio.Compose([
    tio.ZNormalization(),
])
# Definisco padding
padding = tio.CropOrPad(
    (176,256,256),
    only_pad=True
)

# Directory per le immagini preprocessate
root_img = r"C:\Users\marti\Desktop\besta\TLE\TLE_49_56\R1"
output_dir = os.path.join(root_img, 'preprocessed_T1w')  # Dove salvare le immagini trasformate
os.makedirs(output_dir, exist_ok=True)

# Preprocessing di tutte le immagini prima dello split
all_subjects = []
for file in all_files:
    img_path = os.path.join(root_dir, file)
    participant_id = os.path.splitext(file)[0]
    label = label_mapping.get(participant_id, 0)
    subject = tio.Subject(
        image=tio.ScalarImage(img_path),
        label=torch.tensor(label, dtype=torch.long),
        participant_id=participant_id
    )

    # Trasformo le dimensioni per HC e NIGUARDA
    if file.startswith("3TLE_NIGUARDA") or file.startswith("3TLE_HC"):
        # Ottenere il tensore dell'immagine
        tensor = subject['image'].data  # (1, y, z, x) – ordine di assi iniziale
        affine = subject['image'].affine  # Matrice affine

        # Esegui la trasposizione per passare da (1, y, z, x) a (1, x, y, z)
        tensor = tensor.permute(0, 3, 1, 2)  # (1, x, y, z)

        # Crea un nuovo oggetto ScalarImage con il tensore trasposto
        subject['image'] = tio.ScalarImage(tensor=tensor, affine=affine)

        # Padding
        subject=padding(subject)
        
    print(f"Immagine trasformata: {file}: {subject['image'].shape}, Label: {label}")

    # Applica normalizzazione
    subject = preprocess_transform(subject)
    all_subjects.append(subject)

    # Salva immagine trasformata
    output_img_path = os.path.join(output_dir, f"{participant_id}_preprocessed.nii.gz")
    subject['image'].save(output_img_path)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
accuracies = []

for fold, (train_idx, test_idx) in enumerate(kf.split(all_subjects, all_labels)):
    print(f"Training fold {fold+1}/{num_folds}")

    # Split
    train_val_subjects = [all_subjects[i] for i in train_idx]
    test_subjects = [all_subjects[i] for i in test_idx]

    train_val_labels = [all_labels[i] for i in train_idx]

    inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_train_idx, inner_val_idx = next(inner_skf.split(train_idx, train_val_labels))

    train_subjects = [train_val_subjects[i] for i in inner_train_idx]
    val_subjects = [train_val_subjects[i] for i in inner_val_idx]

    # Augmentation solo sul train
    train_transform = tio.RandomAffine(degrees=15)

    # Dataset
    train_dataset = tio.SubjectsDataset(train_subjects, transform=train_transform)
    val_dataset = tio.SubjectsDataset(val_subjects)
    test_dataset = tio.SubjectsDataset(test_subjects)

    # Loader
    train_loader = tio.SubjectsLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = tio.SubjectsLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = tio.SubjectsLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Modello
    model = resnet18(sample_input_D=176, sample_input_H=256, sample_input_W=256, num_seg_classes=1)
    model.to(device)

    # Crea una directory dove salvare il summary, se non esiste
    os.makedirs("model_summaries", exist_ok=True)

    # Salva il summary in un file .txt (solo al primo fold, per evitare ripetizioni)
    if fold == 0:
        with open("model_summaries/resnet18_summary.txt", "w") as f:
            with redirect_stdout(f):
                summary(model, input_size=(1, 176, 256, 256))  # (channels, D, H, W)

    #Caricamento pesi pre-addestrati
    net_dict = model.state_dict()
    pretrain = torch.load(r"C:\Users\marti\Desktop\besta\TLE\TLE_49_56\R1\resnet_18_23dataset.pth", map_location=device)
    pretrain_names = pretrain['state_dict']
    renamed_dict = {k.replace('module.', ''): v for k, v in pretrain_names.items()}
    pretrain_dict = {k: v for k, v in renamed_dict.items() if k in net_dict.keys()}
    net_dict.update(pretrain_dict)
    model.load_state_dict(net_dict)

    # Freezing parziale
    for name, param in model.named_parameters():
        if not (name.startswith("fc") or name.startswith("layer4") or name.startswith("avgpool")):
            param.requires_grad = False

    # Ottimizzazione
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience_lr)

    # Training
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    #torch.cuda.synchronize()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch['image'][tio.DATA].to(device)
            labels = batch['label'].to(device).float().unsqueeze(1)
            participant_ids = batch['participant_id'] 

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Aggiungi il print per mostrare l'output e l'ID dei partecipanti
            print(f"[TRAIN] Loss: {loss.item():.4f}")
            print(f"[TRAIN] Batch Participant IDs: {participant_ids}")


        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'][tio.DATA].to(device)
                labels = batch['label'].to(device).float().unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_fc_bias = model.fc.bias.detach().clone()              # Salva il valore di fc.bias sulla GPU
            torch.save({'state_dict':model.state_dict()}, f"best_model_fold_{fold}.pth")
            
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

    # Crea una cartella per i grafici se non esiste
    os.makedirs('plots', exist_ok=True)

    # Al termine del training del fold, puoi tracciare le perdite per il fold corrente
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss per Epoch - Fold {fold + 1}')
    plt.legend()
    # Salva il grafico in un file (es. PNG)
    plt.savefig(f'plots/loss_fold_{fold + 1}.png', dpi=300)
    #plt.show()

    best_weights = torch.load(f"best_model_fold_{fold}.pth")
    best_names = best_weights['state_dict']
    model.load_state_dict(best_names)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'][tio.DATA].to(device)
            labels = batch['label'].to(device)
            participant_id = batch['participant_id']
            print(f"[TEST] Participant ID: {participant_id}")

            labels = labels.unsqueeze(1)  # Ora labels è di forma (N, 1)

            outputs = model(images)
            predicted = torch.sigmoid(outputs).round()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    accuracies.append(accuracy)
    print(f"Fold {fold+1} Accuracy: {accuracy * 100:.8f}%")

    with open("accuracies.txt", "a") as f:
        f.write(f"Fold {fold+1} Accuracy: {accuracy * 100:.8f}%\n")

# Accuracy media
mean_accuracy = np.mean(accuracies)
print(f"Average Accuracy across {num_folds} folds: {mean_accuracy:.4f}")