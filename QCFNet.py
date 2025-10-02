import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pennylane as qml
import torch.nn.functional as F
import os
import random
import sys
from PIL import Image


class Logger(object):
    def __init__(self, filename='qcfnet_denoising.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class CIFAR10Dataset(Dataset):
    def __init__(self, root, train=True, transform=None, num_samples=1000):
        self.dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=True, transform=transform
        )
        self.transform = transform
        self.num_samples = num_samples
        self.indices = np.random.choice(len(self.dataset), num_samples, replace=False)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img, _ = self.dataset[self.indices[idx]]
        img_gray = torch.mean(img, dim=0, keepdim=True)  
        img_gray = img_gray / 255.0 * np.pi  
        img_gray = img_gray.to(dtype=torch.float32)

        
        gamma_shape = 2.0
        gamma_scale = 1.0 / gamma_shape
        noise = np.random.gamma(shape=gamma_shape, scale=gamma_scale, size=img_gray.shape)
        noisy_img = img_gray * torch.tensor(noise, dtype=torch.float32)  
        noisy_img = torch.clip(noisy_img, 0, np.pi)

        return noisy_img, img_gray


class Sentinel1Dataset(Dataset):
    def __init__(self, root, train=True, transform=None, num_pairs=2000, patch_size=64):
        self.root = root
        self.train = train
        self.patch_size = patch_size
        self.num_pairs = num_pairs
        self.transform = transform
        self.clean_paths = [os.path.join(root, 'clean', f) for f in os.listdir(os.path.join(root, 'clean')) if
                            f.endswith('.png')]
        self.noisy_paths = [os.path.join(root, 'noisy', f) for f in os.listdir(os.path.join(root, 'noisy')) if
                            f.endswith('.png')]
        self.clean_paths = self.clean_paths[:num_pairs]
        self.noisy_paths = self.noisy_paths[:num_pairs]
        split_idx = int(0.8 * num_pairs)
        if self.train:
            self.clean_paths = self.clean_paths[:split_idx]
            self.noisy_paths = self.noisy_paths[:split_idx]
        else:
            self.clean_paths = self.clean_paths[split_idx:]
            self.noisy_paths = self.noisy_paths[split_idx:]

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        clean_img = Image.open(self.clean_paths[idx]).convert('L')
        noisy_img = Image.open(self.noisy_paths[idx]).convert('L')
        if self.transform:
            clean_img = self.transform(clean_img)
            noisy_img = self.transform(noisy_img)
        
        clean_img = clean_img.to(dtype=torch.float32) / 255.0 * np.pi
        noisy_img = noisy_img.to(dtype=torch.float32) / 255.0 * np.pi
        clean_img = clean_img.unsqueeze(0)
        noisy_img = noisy_img.unsqueeze(0)

        return noisy_img, clean_img


def load_dataset(dataset_type, root, batch_size=32):
    if dataset_type == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor()])  
        train_set = CIFAR10Dataset(root, train=True, transform=transform, num_samples=1000)
        val_set = CIFAR10Dataset(root, train=False, transform=transform, num_samples=1000)
    elif dataset_type == 'sentinel1':
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomRotation([0, 90, 180, 270]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor() 
        ])
        train_set = Sentinel1Dataset(root, train=True, transform=transform, num_pairs=2000)
        val_set = Sentinel1Dataset(root, train=False, transform=transform, num_pairs=2000)
    else:
        raise ValueError("Only 'cifar10' or 'sentinel1' supported")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


n_qubits = 4
dev = qml.device('default.qubit', wires=n_qubits)


def dynamic_circuit(inputs, weights):
    if len(inputs) < 16:
        raise ValueError("Input length must be at least 16 (4×4 patch)")
    for i in range(4):
        qml.Hadamard(wires=i)  
        for j in range(4):
            index = i * 4 + j
            if j % 2 == 0:
                qml.RY(inputs[index], wires=i)
            else:
                qml.RZ(inputs[index], wires=i)
    qml.RX(weights[0, 0], wires=0)
    qml.RX(weights[0, 1], wires=1)
    qml.RX(weights[0, 2], wires=2)
    qml.RX(weights[0, 3], wires=3)


def extended_circuit(inputs, weights, circuit_structure):
    dynamic_circuit(inputs, weights)
    rx_param_count = 4
    for i, gate in enumerate(circuit_structure):
        gate_type, _, qubits = gate
        if gate_type == 'CRZ':
            adjusted_param_index = rx_param_count + i
            qml.CRZ(weights[0, adjusted_param_index], wires=qubits)
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


class QuantumCircuitModule(nn.Module):
    def __init__(self, circuit_structure):
        super().__init__()
        num_crz_gates = len(circuit_structure)
        weight_shapes = {"weights": (1, 4 + num_crz_gates)}
        qnode = qml.QNode(
            lambda inputs, weights: extended_circuit(inputs, weights, circuit_structure),
            dev, interface='torch', diff_method="best"
        )
        self.ql1 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.circuit_structure = circuit_structure

    def forward(self, x):
        assert len(x.shape) == 4
        bs, _, h, w = x.shape
        x_lst = []
        for i in range(0, h - 3, 1):
            for j in range(0, w - 3, 1):
                patch = x[:, :, i:i + 4, j:j + 4].flatten(start_dim=1)
                x_lst.append(self.ql1(patch))
        h_out = h - 4 + 1
        w_out = w - 4 + 1
        x = torch.cat(x_lst, dim=1).view(bs, 4, h_out, w_out)
        return x



class ClassicalConvModule(nn.Module):
    def __init__(self, in_channels=1, out_channels=12):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)



class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        feat = self.avg_pool(x).view(b, c)
        mask = self.fc(feat).view(b, c, 1, 1)
        return x * mask


class FeatureConcatenation(nn.Module):
    def forward(self, q_feats, c_feats):
        return torch.cat([q_feats, c_feats], dim=1)


class QuantumClassicalFusion(nn.Module):
    def __init__(self, circuit_structure, c_out=12):
        super().__init__()
        self.quanv = QuantumCircuitModule(circuit_structure)
        self.classical_conv = ClassicalConvModule(out_channels=c_out)
        self.concatenation = FeatureConcatenation()
        self.fusion_conv = nn.Conv2d(4 + c_out, 2 * (4 + c_out), kernel_size=3, stride=1, padding=1)
        self.se = SEBlock(2 * (4 + c_out))
        self.shortcut = nn.Sequential(
            nn.Conv2d(1, 2 * (4 + c_out), kernel_size=1, stride=1),
            nn.ReLU()
        )

    def forward(self, x):
        q_feats = self.quanv(x)
        c_feats = self.classical_conv(x)
        q_feats = F.interpolate(q_feats, size=c_feats.shape[2:], mode='bilinear')
        fused_cat = self.concatenation(q_feats, c_feats)
        fused_conv = self.fusion_conv(fused_cat)
        fused_se = self.se(fused_conv)
        shortcut = self.shortcut(x)
        shortcut = F.interpolate(shortcut, size=fused_se.shape[2:], mode='bilinear')
        return fused_se + shortcut



class DenoisingCNN(nn.Module):
    def __init__(self, circuit_structure, c_out=12, img_size=32):
        super().__init__()
        self.img_size = img_size
        self.fusion_module = QuantumClassicalFusion(circuit_structure, c_out=c_out)
        fusion_out_ch = 2 * (4 + c_out)

        self.encoder = nn.Sequential(
            nn.Conv2d(fusion_out_ch, 2 * fusion_out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * fusion_out_ch, 4 * fusion_out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4 * fusion_out_ch, 2 * fusion_out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * fusion_out_ch, fusion_out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(fusion_out_ch, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, noisy_img):
        fused_feats = self.fusion_module(noisy_img)
        encoded = self.encoder(fused_feats)
        decoded = self.decoder(encoded)
        decoded = F.interpolate(decoded, size=(self.img_size, self.img_size), mode='bilinear')
        decoded = decoded * np.pi
        return decoded



class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
        self.vgg = torchvision.models.vgg16(weights=weights).features[:16].eval()
        for p in self.vgg.parameters():
            p.requires_grad = False

    def forward(self, pred, target):
        pred = F.interpolate(pred, size=(224, 224), mode='bilinear')
        target = F.interpolate(target, size=(224, 224), mode='bilinear')
        pred = pred.repeat(1, 3, 1, 1)
        target = target.repeat(1, 3, 1, 1)
        pred_feat = self.vgg(pred)
        target_feat = self.vgg(target)
        return F.mse_loss(pred_feat, target_feat)


class AdaptiveLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.001, gamma=0.5, epsilon=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()

    def forward(self, output, target):
        mse = self.mse_loss(output, target)
        output_np = output.squeeze(1).cpu().detach().numpy()
        target_np = target.squeeze(1).cpu().detach().numpy()
        ssim_val = np.mean([ssim(o, t, data_range=np.pi) for o, t in zip(output_np, target_np)])
        ssim_loss = 1 - ssim_val
        ssim_loss = torch.tensor(ssim_loss, dtype=torch.float32, device=output.device) 
        perceptual_loss = self.perceptual_loss(output, target)

        mse_weight = 1 / (mse + self.epsilon)
        ssim_weight = 1 / (ssim_loss + self.epsilon)
        perceptual_weight = 1 / (perceptual_loss + self.epsilon)
        total_weight = mse_weight + ssim_weight + perceptual_weight
        mse_weight /= total_weight
        ssim_weight /= total_weight
        perceptual_weight /= total_weight

        loss = (
                self.alpha * mse_weight * mse +
                self.beta * ssim_weight * ssim_loss +
                self.gamma * perceptual_weight * perceptual_loss
        )
        return loss



class MCTSNode:
    def __init__(self, circuit_structure, parent=None, max_gates=4, c_decay=0.999):
        self.circuit_structure = circuit_structure
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_reward = 0
        self.max_gates = max_gates
        self.c_decay = c_decay

    def _get_available_gates(self):
        all_crz = [('CRZ', idx, (c, t)) for idx, (c, t) in enumerate(
            [(c, t) for c in range(4) for t in range(4) if c != t]
        )]
        used_gates = set(tuple(g) for g in self.circuit_structure)
        used_target = set(g[2][1] for g in self.circuit_structure)
        available = [g for g in all_crz if tuple(g) not in used_gates and g[2][1] not in used_target]
        return available

    def expand(self):
        if len(self.circuit_structure) >= self.max_gates:
            return []
        available_gates = self._get_available_gates()
        self.children = [
            MCTSNode(self.circuit_structure + [gate], self, self.max_gates)
            for gate in available_gates
        ]
        return self.children

    def uct_value(self):
        if self.visits == 0:
            return float('inf')
        parent_visits = self.parent.visits if self.parent else 1
        c = 1.4 * (self.c_decay ** parent_visits)
        exploitation = self.total_reward / self.visits
        exploration = c * np.sqrt(np.log(parent_visits) / self.visits)
        return exploitation + exploration

    def simulate(self, val_loader, device, pre_trained_classical_params):
        full_circuit = self.circuit_structure
        img_size = val_loader.dataset[0][0].shape[2]
        model = DenoisingCNN(full_circuit, img_size=img_size)
        model.to(device)

        classical_params = {k: v for k, v in pre_trained_classical_params.items() if 'quanv' not in k}
        model.load_state_dict(classical_params, strict=False)

        quantum_params = [p for n, p in model.named_parameters() if 'quanv' in n]
        optimizer = optim.Adam(quantum_params, lr=0.01)
        criterion = AdaptiveLoss()
        model.train()
        for _ in range(3):
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                optimizer.zero_grad()
                pred = model(noisy)
                loss = criterion(pred, clean)
                loss.backward()
                optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                pred = model(noisy)
                val_loss += criterion(pred, clean).item()
        val_loss /= len(val_loader)
        return -val_loss

    def backpropagate(self, reward):
        node = self
        while node:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def is_leaf(self):
        return len(self.circuit_structure) >= self.max_gates or len(self._get_available_gates()) == 0


def monte_carlo_tree_search(train_loader, val_loader, max_gates=4, num_simulations=2000, device="cpu"):
    print("Pre-training classical module...")
    img_size = train_loader.dataset[0][0].shape[2]
    pre_train_model = DenoisingCNN(circuit_structure=[], img_size=img_size)
    pre_train_model.to(device)

    pre_train_optimizer = optim.Adam(pre_train_model.parameters(), lr=0.001)
    pre_train_criterion = AdaptiveLoss()
    for epoch in range(10):
        pre_train_model.train()
        total_loss = 0.0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            
            random_crz = random.sample([('CRZ', 0, (c, t)) for c in range(4) for t in range(4) if c != t],
                                       k=random.randint(1, 4))
            pre_train_model.fusion_module.quanv = QuantumCircuitModule(random_crz)
            pre_train_optimizer.zero_grad()
            pred = pre_train_model(noisy)
            loss = pre_train_criterion(pred, clean)
            loss.backward()
            pre_train_optimizer.step()
            total_loss += loss.item()
        print(f"Pre-train Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")
    pre_trained_classical_params = pre_train_model.state_dict()

    root = MCTSNode(circuit_structure=[], max_gates=max_gates)
    for sim in range(num_simulations):
        if sim % 200 == 0:
            print(f"MCTS Simulation {sim + 1}/{num_simulations}")

        current_node = root
        while not current_node.is_leaf():
            if not current_node.children:
                current_node.expand()
            current_node = max(current_node.children, key=lambda n: n.uct_value())

        if not current_node.is_leaf():
            current_node.expand()
            current_node = current_node.children[0]

        reward = current_node.simulate(val_loader, device, pre_trained_classical_params)
        current_node.backpropagate(reward)

    def get_best_node(node):
        if not node.children:
            return node
        best_child = max(node.children, key=lambda n: n.total_reward / n.visits if n.visits else -float('inf'))
        return get_best_node(best_child)

    best_node = get_best_node(root)
    print(f"Best Quantum Circuit Structure: {best_node.circuit_structure}")
    return best_node.circuit_structure



def train_model(model, train_loader, val_loader, epochs=20, lr=0.001, device="cpu"):
    quantum_params = [p for n, p in model.named_parameters() if 'quanv' in n]
    classical_params = [p for n, p in model.named_parameters() if 'quanv' not in n]
    optimizer = optim.Adam([
        {'params': quantum_params, 'lr': lr * 10},
        {'params': classical_params, 'lr': lr}
    ])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = AdaptiveLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for noisy_imgs, clean_imgs in train_loader:
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            optimizer.zero_grad()
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy_imgs, clean_imgs in val_loader:
                noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
                outputs = model(noisy_imgs)
                loss = criterion(outputs, clean_imgs)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        scheduler.step()
    return model


def evaluate_model(model, val_loader, device="cpu"):
    model.eval()
    model.to(device)
    mse_list, ssim_list, psnr_list = [], [], []
    with torch.no_grad():
        for noisy_imgs, clean_imgs in val_loader:
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            outputs = model(noisy_imgs)
            output_np = outputs.squeeze(1).cpu().numpy()
            target_np = clean_imgs.squeeze(1).cpu().numpy()
            for o, t in zip(output_np, target_np):
                mse_list.append(np.mean((o - t) ** 2))
                ssim_list.append(ssim(o, t, data_range=np.pi))
                psnr_list.append(psnr(o, t, data_range=np.pi))
    mse = np.mean(mse_list)
    ssim_val = np.mean(ssim_list)
    psnr_val = np.mean(psnr_list)
    print(f"Evaluation Results - MSE: {mse:.4f}, SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.4f}")
    return mse, ssim_val, psnr_val



if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    sys.stdout = Logger('qcfnet_denoising.log', sys.stdout)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset_type = 'cifar10'
    if dataset_type == 'cifar10':
        train_loader, val_loader = load_dataset('cifar10', root='./data/cifar10', batch_size=32)
    else:
        train_loader, val_loader = load_dataset('sentinel1', root='./data/sentinel1', batch_size=32)

    best_circuit = monte_carlo_tree_search(
        train_loader, val_loader, max_gates=4, num_simulations=2000, device=device
    )

    print("Training best model...")
    img_size = train_loader.dataset[0][0].shape[2]
    best_model = DenoisingCNN(best_circuit, img_size=img_size)
    best_model = train_model(best_model, train_loader, val_loader, epochs=20, lr=0.001, device=device)

    torch.save(best_model.state_dict(), f"qcfnet_best_{dataset_type}.pth")
    print(f"Model saved as qcfnet_best_{dataset_type}.pth")

    evaluate_model(best_model, val_loader, device=device)

    sys.stdout.log.close()
    sys.stdout = sys.__stdout__

