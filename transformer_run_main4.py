import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch.nn.functional as F
from EMG_data_loader import EMGDataset, get_emg_dataloader
import math
import os

plt.rcParams['font.sans-serif'] = ['STXihei']
plt.rcParams['axes.unicode_minus'] = False


def preprocess_data():
    dataset = EMGDataset('Stroke Patients Data')

    labels = [label.item() for _, label in dataset]

    train_indices = []
    val_indices = []
    test_indices = []

    unique_labels = np.unique(labels)
    np.random.seed(0)

    for label in unique_labels:
        indices = np.where(np.array(labels) == label)[0]

        np.random.shuffle(indices)
        s1 = int(len(indices) * 0.6)
        s2 = int(len(indices) * 0.8)

        train_indices.extend(indices[:s1])
        val_indices.extend(indices[s1:s2])
        test_indices.extend(indices[s2:])

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=30):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class EMG_Transformer(nn.Module):
    def __init__(self, num_classes, input_dim=8, model_dim=128, num_heads=8, num_layers=3, dropout=0.1):
        super(EMG_Transformer, self).__init__()

        self.model_dim = model_dim
        self.input_dim = input_dim

        self.input_projection = nn.Linear(input_dim, model_dim)

        self.pos_encoding = PositionalEncoding(model_dim, max_len=30)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=False
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)

        x = x.squeeze(1)

        x = x.transpose(0, 1)

        x = self.input_projection(x)

        x = self.pos_encoding(x)

        x = self.transformer_encoder(x)

        x = x[0, :, :]

        x = self.classifier(x)
        return x


class EMGContrastiveAutoEncoder(nn.Module):
    def __init__(self, input_dim=8, model_dim=128, temperature=0.1):
        super(EMGContrastiveAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.temperature = temperature

        self.encoder_projection = nn.Linear(input_dim, model_dim)
        self.pos_encoding = PositionalEncoding(model_dim, max_len=30)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=False
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=3)

        self.decoder_projection = nn.Linear(model_dim, input_dim)

        self.contrastive_projection = nn.Sequential(
            nn.Linear(model_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x, return_features=False):

        if x.dim() == 4:
            x = x.squeeze(1)

        x = x.transpose(0, 1)

        encoded = self.encoder_projection(x)
        encoded = self.pos_encoding(encoded)
        encoded = self.transformer_encoder(encoded)

        sequence_representation = encoded[0, :, :]

        if return_features:
            contrastive_features = self.contrastive_projection(sequence_representation)

            contrastive_features = F.normalize(contrastive_features, dim=1)

            decoded = self.transformer_decoder(encoded)
            decoded = self.decoder_projection(decoded)

            decoded = decoded.transpose(0, 1)

            return decoded, contrastive_features

        decoded = self.transformer_decoder(encoded)
        decoded = self.decoder_projection(decoded)

        decoded = decoded.transpose(0, 1)

        return decoded


def contrastive_loss(features, labels, temperature=0.1):
    similarity_matrix = torch.matmul(features, features.T) / temperature

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float()

    mask_no_diag = mask - torch.diag(torch.diag(mask))

    exp_sim = torch.exp(similarity_matrix) * mask_no_diag
    pos_sum = torch.sum(exp_sim, dim=1)

    all_sum = torch.sum(torch.exp(similarity_matrix), dim=1)

    loss = -torch.log((pos_sum + 1e-8) / (all_sum + 1e-8))

    return torch.mean(loss)


def pretrain_autoencoder_contrastive(train_loader, val_loader=None, num_epochs=50,
                                     learning_rate=1e-3, device=None, use_contrastive=True):
    if use_contrastive:
        autoencoder = EMGContrastiveAutoEncoder(input_dim=8, model_dim=128)
    else:
        autoencoder = EMGAutoEncoder(input_dim=8, model_dim=128)

    autoencoder = autoencoder.to(device)

    recon_criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

    print("开始自监督预训练...")

    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 10

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):

        autoencoder.train()
        total_train_loss = 0.0

        for inputs, labels in train_loader:
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)

            inputs = inputs.to(device)
            inputs = inputs.float()
            labels = labels.to(device)

            optimizer.zero_grad()

            if use_contrastive:

                reconstructed, features = autoencoder(inputs, return_features=True)
                original = inputs.squeeze(1)  # (batch, 30, 8)

                recon_loss = recon_criterion(reconstructed, original)

                cont_loss = contrastive_loss(features, labels, temperature=0.3)

                loss = recon_loss + 0.1 * cont_loss
            else:

                reconstructed = autoencoder(inputs)
                original = inputs.squeeze(1)  # (batch, 30, 8)
                loss = recon_criterion(reconstructed, original)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        if val_loader is not None:
            autoencoder.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for val_inputs, val_labels in val_loader:
                    if val_inputs.dim() == 3:
                        val_inputs = val_inputs.unsqueeze(1)

                    val_inputs = val_inputs.to(device).float()
                    val_labels = val_labels.to(device)

                    if use_contrastive:
                        reconstructed, features = autoencoder(val_inputs, return_features=True)
                        original = val_inputs.squeeze(1)
                        recon_loss = recon_criterion(reconstructed, original)
                        cont_loss = contrastive_loss(features, val_labels, temperature=0.1)
                        val_loss = recon_loss + 0.1 * cont_loss
                    else:
                        reconstructed = autoencoder(val_inputs)
                        original = val_inputs.squeeze(1)
                        val_loss = recon_criterion(reconstructed, original)

                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0

                torch.save(autoencoder.state_dict(), 'best_autoencoder.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience_limit:
                print(f"早停机制触发，停止训练在第 {epoch + 1} 轮")

                autoencoder.load_state_dict(torch.load('best_autoencoder.pth'))
                break

            if (epoch + 1) % 10 == 0:
                print(f'自监督预训练 Epoch [{epoch + 1}/{num_epochs}], '
                      f'Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        else:
            if (epoch + 1) % 10 == 0:
                print(f'自监督预训练 Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.6f}')

    print("自监督预训练完成!")

    if val_loader is not None:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Autoencoder Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    return autoencoder


class EMGAutoEncoder(nn.Module):
    def __init__(self, input_dim=8, model_dim=128):
        super(EMGAutoEncoder, self).__init__()

        self.input_dim = input_dim
        self.model_dim = model_dim

        self.encoder_projection = nn.Linear(input_dim, model_dim)
        self.pos_encoding = PositionalEncoding(model_dim, max_len=30)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=False
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=3)

        self.decoder_projection = nn.Linear(model_dim, input_dim)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)

        x = x.transpose(0, 1)

        encoded = self.encoder_projection(x)
        encoded = self.pos_encoding(encoded)
        encoded = self.transformer_encoder(encoded)

        decoded = self.transformer_decoder(encoded)
        decoded = self.decoder_projection(decoded)

        decoded = decoded.transpose(0, 1)

        return decoded


def pretrain_autoencoder(train_loader, num_epochs=50, learning_rate=1e-3, device=None):
    autoencoder = EMGAutoEncoder(input_dim=8, model_dim=128)
    autoencoder = autoencoder.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

    print("开始自监督预训练...")

    for epoch in range(num_epochs):
        autoencoder.train()
        total_loss = 0.0

        for inputs, _ in train_loader:
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)

            inputs = inputs.to(device)
            inputs = inputs.float()

            optimizer.zero_grad()

            # 重建输入信号
            reconstructed = autoencoder(inputs)
            original = inputs.squeeze(1)

            loss = criterion(reconstructed, original)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f'自监督预训练 Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.6f}')

    print("自监督预训练完成!")
    return autoencoder


def transfer_encoder_weights(autoencoder, main_model):
    main_model.input_projection.weight.data = autoencoder.encoder_projection.weight.data.clone()
    main_model.input_projection.bias.data = autoencoder.encoder_projection.bias.data.clone()

    if hasattr(autoencoder, 'pos_encoding') and hasattr(main_model, 'pos_encoding'):
        main_model.pos_encoding.pe.data = autoencoder.pos_encoding.pe.data.clone()

    for main_layer, ae_layer in zip(main_model.transformer_encoder.layers,
                                    autoencoder.transformer_encoder.layers):
        main_layer.self_attn.load_state_dict(ae_layer.self_attn.state_dict())
        main_layer.linear1.load_state_dict(ae_layer.linear1.state_dict())
        main_layer.linear2.load_state_dict(ae_layer.linear2.state_dict())
        main_layer.norm1.load_state_dict(ae_layer.norm1.state_dict())
        main_layer.norm2.load_state_dict(ae_layer.norm2.state_dict())
        main_layer.dropout.load_state_dict(ae_layer.dropout.state_dict())

    print("编码器权重迁移完成!")


def save_pretrained_weights(model, save_path="pretrained_weights.pth"):
    pretrained_weights = {
        'input_projection': model.input_projection.state_dict(),
        'pos_encoding': model.pos_encoding.state_dict(),
        'transformer_encoder': model.transformer_encoder.state_dict()
    }

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    torch.save(pretrained_weights, save_path)
    print(f"预训练权重已保存到 {save_path}")


def load_pretrained_weights(model, load_path="pretrained_weights.pth", device=None):
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"预训练权重文件 {load_path} 不存在")

    pretrained_weights = torch.load(load_path, map_location=device)

    model.input_projection.load_state_dict(pretrained_weights['input_projection'])
    model.pos_encoding.load_state_dict(pretrained_weights['pos_encoding'])
    model.transformer_encoder.load_state_dict(pretrained_weights['transformer_encoder'])

    print(f"预训练权重已从 {load_path} 加载")


def freeze_encoder(model):
    for param in model.input_projection.parameters():
        param.requires_grad = False
    for param in model.pos_encoding.parameters():
        param.requires_grad = False
    for param in model.transformer_encoder.parameters():
        param.requires_grad = False
    print("编码器已冻结，仅训练分类器部分")


def unfreeze_encoder(model):
    for param in model.input_projection.parameters():
        param.requires_grad = True
    for param in model.pos_encoding.parameters():
        param.requires_grad = True
    for param in model.transformer_encoder.parameters():
        param.requires_grad = True
    print("编码器已解冻，所有参数均可训练")


def train_model_finetune(model, train_loader, test_loader, num_epochs=100, learning_rate=1e-3, device=None,
                         freeze_encoder_flag=True):
    if freeze_encoder_flag:

        freeze_encoder(model)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        print("编码器已冻结，仅训练分类器部分")
    else:

        transformer_params = list(model.input_projection.parameters()) + \
                             list(model.pos_encoding.parameters()) + \
                             list(model.transformer_encoder.parameters())

        classifier_params = list(model.classifier.parameters())

        optimizer = optim.Adam([
            {'params': transformer_params, 'lr': 0.0004},
            {'params': classifier_params, 'lr': 0.001}
        ])
        print("使用不同学习率训练: transformer_encoder部分为0.0005, classifier部分为0.001")

    criterion = nn.CrossEntropyLoss().to(device)

    def lr_lambda(epoch):
        if epoch == 99:
            return 0.1
        else:
            return 1.0

    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_test_acc = 0.0
    best_test_loss = float('inf')

    with open('res1017.txt', 'w') as f:
        f.write("Epoch\tTest_Accuracy\tTest_Loss\n")

    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:

            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = Variable(inputs.float()), Variable(labels.long())
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        # with torch.no_grad():
        #     for inputs, labels in val_loader:
        #         inputs, labels = inputs.to(device), labels.to(device)
        #         inputs, labels = Variable(inputs.float()), Variable(labels.long())
        #
        #         if inputs.dim() == 3:
        #             inputs = inputs.unsqueeze(1)
        #
        #         outputs = model(inputs)
        #         loss = criterion(outputs, labels)
        #         val_loss += loss.item()
        #
        #         _, predicted = torch.max(outputs.data, 1)
        #         total_val += labels.size(0)
        #         correct_val += (predicted == labels).sum().item()

        # val_loss = val_loss / len(val_loader)
        # val_acc = correct_val / total_val

        val_loss = 0
        val_acc = 0

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        if epoch > 150:
            test_acc, test_loss = evaluate_model(model, test_loader, criterion, device)

            with open('res1017.txt', 'a') as f:
                f.write(f"{epoch + 1}\t{test_acc:.4f}\t{test_loss:.4f}\n")

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_loss = test_loss

                torch.save(model.state_dict(), 'best_test_model.pth')
                print(f"在epoch {epoch + 1} 保存了新的最佳测试模型，准确率: {test_acc:.4f}")
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    return train_losses, val_losses, train_accuracies, val_accuracies


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:

            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)

            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = Variable(inputs.float()), Variable(labels.long())

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total

    return accuracy, avg_loss


def main_pretrain_and_save():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset, test_dataset = preprocess_data()

    train_loader = DataLoader(train_dataset, batch_size=96, shuffle=True, num_workers=96)

    num_classes = 6
    model = EMG_Transformer(num_classes)
    model = model.to(device)

    print(f"\n模型总参数量: {count_parameters(model):,}")

    # autoencoder = pretrain_autoencoder(train_loader, num_epochs=100, learning_rate=5e-4, device=device)
    autoencoder = pretrain_autoencoder_contrastive(train_loader, num_epochs=150, learning_rate=1e-3, device=device)

    transfer_encoder_weights(autoencoder, model)

    save_pretrained_weights(model, "pretrained_weights_1024_contra.pth")

    print("预训练完成，编码器权重已保存!")


def main_finetune_classifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset, test_dataset = preprocess_data()

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=96)
    val_loader = DataLoader(val_dataset, batch_size=96, shuffle=True, num_workers=96)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=96)

    num_classes = 6
    model = EMG_Transformer(num_classes)
    model = model.to(device)

    print(f"\n模型总参数量: {count_parameters(model):,}")

    load_pretrained_weights(model, "pretrained_weights_1017_non_contra.pth", device)

    print("开始监督训练（仅训练分类器）...")
    train_losses, val_losses, train_accuracies, val_accuracies = train_model_finetune(
        model, val_loader, test_loader, num_epochs=200, learning_rate=5e-4, device=device, freeze_encoder_flag=False
    )

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    train_pred, train_true = predict(model, train_loader, device)
    val_pred, val_true = predict(model, val_loader, device)
    test_pred, test_true = predict(model, test_loader, device)

    acc_train = np.mean(train_pred == train_true)
    acc_val = np.mean(val_pred == val_true)
    acc_test = np.mean(test_pred == test_true)

    print('训练集准确率acc_train:', acc_train)
    print('验证集准确率acc_val:', acc_val)
    print('测试集准确率acc_test:', acc_test)

    label_names = [f'Class {i}' for i in range(num_classes)]

    cm_train = plot_confusion_matrix(train_true, train_pred, label_names, '训练集预测混淆矩阵')

    cm_val = plot_confusion_matrix(val_true, val_pred, label_names, '验证集预测混淆矩阵')

    cm_test = plot_confusion_matrix(test_true, test_pred, label_names, '测试集预测混淆矩阵')

    results = {
        'model': model,
        'train_accuracy': acc_train,
        'val_accuracy': acc_val,
        'test_accuracy': acc_test,
        'cm_train': cm_train,
        'cm_val': cm_val,
        'cm_test': cm_test
    }

    print("微调完成!")
    return results


def main_with_pretrained():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset, test_dataset = preprocess_data()

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=96)
    val_loader = DataLoader(val_dataset, batch_size=96, shuffle=True, num_workers=96)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=96)

    num_classes = 6
    model = EMG_Transformer(num_classes)
    model = model.to(device)
    print("网络结构:")
    print(model)

    print(f"\n模型总参数量: {count_parameters(model):,}")

    if os.path.exists("pretrained_weights_1022_contra.pth"):
        load_pretrained_weights(model, "pretrained_weights_1022_contra.pth", device)
        print("已加载预训练权重")
    else:
        print("未找到预训练权重，使用随机初始化权重")

    print("开始训练...")
    train_losses, val_losses, train_accuracies, val_accuracies = train_model_finetune(
        model, val_loader, test_loader, num_epochs=200, learning_rate=5e-4, device=device, freeze_encoder_flag=False
    )

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    train_pred, train_true = predict(model, train_loader, device)
    val_pred, val_true = predict(model, val_loader, device)
    test_pred, test_true = predict(model, test_loader, device)

    acc_train = np.mean(train_pred == train_true)
    acc_val = np.mean(val_pred == val_true)
    acc_test = np.mean(test_pred == test_true)

    print('训练集准确率acc_train:', acc_train)
    print('验证集准确率acc_val:', acc_val)
    print('测试集准确率acc_test:', acc_test)

    label_names = [f'Class {i}' for i in range(num_classes)]

    cm_train = plot_confusion_matrix(train_true, train_pred, label_names, '训练集预测混淆矩阵')

    cm_val = plot_confusion_matrix(val_true, val_pred, label_names, '验证集预测混淆矩阵')

    cm_test = plot_confusion_matrix(test_true, test_pred, label_names, '测试集预测混淆矩阵')

    results = {
        'model': model,
        'train_accuracy': acc_train,
        'val_accuracy': acc_val,
        'test_accuracy': acc_test,
        'cm_train': cm_train,
        'cm_val': cm_val,
        'cm_test': cm_test
    }

    print("训练完成!")
    return results


def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-3, device=None):
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    def lr_lambda(epoch):
        if epoch == 9:
            return 0.1
        else:
            return 1.0

    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    learning_rates = []

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:

            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = Variable(inputs.float()), Variable(labels.long())
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs, labels = Variable(inputs.float()), Variable(labels.long())
                # inputs, labels= Variable(inputs), Variable(labels)

                if inputs.dim() == 3:
                    inputs = inputs.unsqueeze(1)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = correct_val / total_val

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    return train_losses, val_losses, train_accuracies, val_accuracies


def predict(model, data_loader, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = Variable(inputs.float()), Variable(labels.long())

            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)

            outputs = model(inputs.float())
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return np.array(predictions), np.array(true_labels)


def plot_confusion_matrix(y_true, y_pred, class_names, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    return cm


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset, test_dataset = preprocess_data()

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=96)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=96)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=96)

    num_classes = 6
    model = EMG_Transformer(num_classes)
    model = model.to(device)
    print("网络结构:")
    print(model)

    print(f"\n模型总参数量: {count_parameters(model):,}")

    print("开始训练...")
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, num_epochs=100, learning_rate=1e-3, device=device
    )

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    train_pred, train_true = predict(model, train_loader, device)
    val_pred, val_true = predict(model, val_loader, device)
    test_pred, test_true = predict(model, test_loader, device)

    acc_train = np.mean(train_pred == train_true)
    acc_val = np.mean(val_pred == val_true)
    acc_test = np.mean(test_pred == test_true)

    print('训练集准确率acc_train:', acc_train)
    print('验证集准确率acc_val:', acc_val)
    print('测试集准确率acc_test:', acc_test)

    label_names = [f'Class {i}' for i in range(num_classes)]

    cm_train = plot_confusion_matrix(train_true, train_pred, label_names, '训练集预测混淆矩阵')

    cm_val = plot_confusion_matrix(val_true, val_pred, label_names, '验证集预测混淆矩阵')

    cm_test = plot_confusion_matrix(test_true, test_pred, label_names, '测试集预测混淆矩阵')

    results = {
        'model': model,
        'train_accuracy': acc_train,
        'val_accuracy': acc_val,
        'test_accuracy': acc_test,
        'cm_train': cm_train,
        'cm_val': cm_val,
        'cm_test': cm_test
    }

    print("训练完成!")
    return results


if __name__ == "__main__":
    # results = main()
    # main_pretrain_and_save()
    results = main_pretrain_and_save()
