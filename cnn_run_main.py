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

plt.rcParams['font.sans-serif'] = ['STXihei']
plt.rcParams['axes.unicode_minus'] = False


# 数据预处理函数
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


class EMG_CNN(nn.Module):
    def __init__(self, num_classes):
        super(EMG_CNN, self).__init__()
        self.features = nn.Sequential(

            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 15 * 4, 256),
            nn.Linear(256, 512),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_model(model, train_loader, val_loader, test_loader, num_epochs=10, learning_rate=1e-3, device=None):
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

    with open('res.txt', 'w') as f:
        f.write("Epoch\tTest_Accuracy\tTest_Loss\n")

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

        if epoch > 150:
            test_acc, test_loss = evaluate_model(model, test_loader, criterion, device)

            with open('res.txt', 'a') as f:
                f.write(f"{epoch + 1}\t{test_acc:.4f}\t{test_loss:.4f}\n")

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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset, test_dataset = preprocess_data()

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=96)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=96)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=96)

    num_classes = 6
    model = EMG_CNN(num_classes)
    model = model.to(device)
    print("网络结构:")
    print(model)

    print(f"\n模型总参数量: {count_parameters(model):,}")

    print("开始训练...")
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, test_loader, num_epochs=200, learning_rate=1e-3, device=device
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
    results = main()
