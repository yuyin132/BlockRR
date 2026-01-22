import logging
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import wandb
import os
import datetime

from tqdm import tqdm
from utils.augmentation import mixup_data, rand_aug_cifar10, rand_aug_cifar100, rand_aug_mnist
from utils.report_log import init_logger
from models import PreActResNet18
from datasets import make_dataset, MyTensorDataset
from randomized import compute_randomized_labels_classification, compute_randomized_labels_classification_infty




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
 
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
 
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0



def get_rand_aug(dataset: str):
    if dataset in ['cifar10_1', 'cifar10_2']:
        train_transform = transforms.Compose(rand_aug_cifar10)
    elif dataset == 'cifar100':
        train_transform = transforms.Compose(rand_aug_cifar100)
    elif dataset == 'mnist':
        train_transform = transforms.Compose(rand_aug_mnist)
    return train_transform


class SymmetricCrossEntropyLoss1(torch.nn.Module):
    # hard label sce loss
    def __init__(self, alpha=1.0, beta=1.0, num_classes=10):
        super(SymmetricCrossEntropyLoss1, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.ce = torch.nn.CrossEntropyLoss()
        
    def forward(self, pred, target):
        ce_loss = self.ce(pred,target)

        pred_softmax = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target.long(), self.num_classes).float()
        rce_loss = (- pred_softmax * torch.log(target_one_hot+1e-4)).sum(dim=1).mean()

        loss = self.alpha * ce_loss + self.beta * rce_loss
        
        return loss



class SymmetricCrossEntropyLoss2(torch.nn.Module):
    # soft label sce loss
    def __init__(self, alpha=1.0, beta=1.0):
        super(SymmetricCrossEntropyLoss2, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, preds, targets):
        """
        Args:
            preds: logits, shape (B, C)
            targets: soft labels, shape (B, C)
        """
        pred_probs = F.softmax(preds, dim=1)
        log_preds = torch.log(pred_probs + 1e-4)
        log_targets = torch.log(targets + 1e-4)

        ce = -(targets * log_preds).sum(dim=1).mean()
        rce = -(pred_probs * log_targets).sum(dim=1).mean()

        return self.alpha * ce + self.beta * rce

def lr_schedule(epoch, total_epochs, warmup_epochs=30):
    warmup_epochs = int(total_epochs * 0.15)
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch + 1) / total_epochs
    if progress < 0.3:
        return 1.0
    elif progress < 0.6:
        return 0.1
    elif progress < 0.9:
        return 0.01
    else:
        return 0.001
    


def create_model(arch: str, num_classes: int):
    if "resnet18v1" in arch.lower():
        print("Created Wide Resnet Model!")
        # return resnet18(num_classes=num_classes)
    else:
        logging.info(f"Created simple Resnet Model!")
        # return models.resnet18(num_classes=num_classes)
        return PreActResNet18(num_classes=num_classes)



def make_deterministic(seed):
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def run_single_experiment(args):
    logging.info(f"Running experiment with:")
    logging.info(f"Task: {args.task}")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Mechanism: {args.mechanism}")
    logging.info(f"Epsilon: {args.epsilon}")
    logging.info(f"Sigma: {args.sigma}")
    logging.info(f"Delta: {args.delta}")
    logging.info(f"Epoch: {args.epoch}")
    logging.info(f"Learning rate: {args.lr}")
    logging.info(f"Batch_size: {args.batch_size}")
    logging.info(f"Weight_decay: {args.weight_decay}")
    logging.info(f"Beta: {args.beta}")
    logging.info(f"Alpha1: {args.alpha1}")
    logging.info(f'Kfolds: {args.kfolds}')

    make_deterministic(args.seed)

    estimate_loader, train_loader, train_rr_loader, test_loader, num_classes = make_dataset(args.dataset, args.task, args.data_dir, args.batch_size, args.seed, args.kfolds)

    if args.epsilon <= 8.0:
        if args.mechanism == 'rr':
            x_sets, y_sets, y_tilde_sets = compute_randomized_labels_classification(train_rr_loader, estimate_loader, args.mechanism, args.prior,
                                                                                num_classes, args.seed, args.epsilon, args.epsilon_p, args.sigma, args.delta)
        else:
            x_sets, y_sets, y_tilde_sets = compute_randomized_labels_classification(train_loader, estimate_loader, args.mechanism, args.prior,
                                                                                    num_classes, args.seed, args.epsilon, args.epsilon_p, args.sigma, args.delta)
    else:
        if args.mechanism == 'rr':
            x_sets, y_sets, y_tilde_sets = compute_randomized_labels_classification_infty(train_rr_loader, estimate_loader, args.mechanism, args.prior,
                                                                                num_classes, args.seed, args.epsilon, args.epsilon_p, args.sigma, args.delta)
        else:
            x_sets, y_sets, y_tilde_sets = compute_randomized_labels_classification_infty(train_loader, estimate_loader, args.mechanism, args.prior,
                                                                                    num_classes, args.seed, args.epsilon, args.epsilon_p, args.sigma, args.delta)
    for i in range(num_classes):
        print(i, (y_tilde_sets == i).sum())
    train_transform = get_rand_aug(args.dataset)
    labeldp_dataset = MyTensorDataset(x_sets.detach(), y_tilde_sets.flatten().detach(), transform=train_transform)
    labeldp_loader = torch.utils.data.DataLoader(labeldp_dataset,
                batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(args.arch, num_classes).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr = args.lr,
        momentum = args.momentum,
        weight_decay = args.weight_decay,
        nesterov = True
        )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_schedule(epoch, args.epoch))
    loss_func1 = SymmetricCrossEntropyLoss1(args.alpha, args.beta, num_classes)
    loss_func2 = SymmetricCrossEntropyLoss2(args.alpha, args.beta)

    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []
    train_acc_label_list = []
    test_acc_label_list = []
    for epoch in range(args.epoch):
        losses = AverageMeter()
        model.train()
        for j, (x, y) in enumerate(labeldp_loader):
                x, y = x.to(device), y.to(device)
                
                onehot_labels = F.one_hot(y, num_classes=num_classes).float()
                mixed_images, mixed_labels = mixup_data(x, onehot_labels, args.alpha1)
                optimizer.zero_grad()
                output = model(mixed_images)       
                loss = loss_func2(output, mixed_labels)  
                loss.backward()        
                optimizer.step()
                losses.update(loss.item(), x.size(0))

        lr_scheduler.step()

        if (epoch+1) % 10 == 0 or args.epoch-epoch <=10 : # (epoch+1) >= 50 and 
                test_loss = AverageMeter()
                train_loss = AverageMeter()
                train_accuracy = AverageMeter()
                test_accuracy = AverageMeter()
                test_accuracy_label = torch.zeros(num_classes)
                test_num_label = torch.zeros(num_classes)
                train_accuracy_label = torch.zeros(num_classes)
                train_num_label = torch.zeros(num_classes)
                true_pos = torch.zeros(num_classes)
                true_neg = torch.zeros(num_classes)
                false_pos = torch.zeros(num_classes)
                false_neg = torch.zeros(num_classes)
                model.eval()
                with torch.no_grad():
                    if args.mechanism == 'rr':
                        for x, y in train_rr_loader:
                                x, y = x.to(device), y.to(device)
                                output = model(x)
                                loss = loss_func1(output, y) 
                                preds = output.argmax(dim=1)
                                correct = (preds == y)
                                y_unique = torch.unique(y)
                                for i in range(len(y_unique)):
                                    train_accuracy_label[y_unique[i]] += correct[y == y_unique[i]].sum().item()
                                    train_num_label[y_unique[i]] += (y==y_unique[i]).sum().item()
                                acc = correct.sum().item() / x.size(0)

                                train_loss.update(loss.item(), x.size(0))
                                train_accuracy.update(acc, x.size(0))
                    else:
                        for x, y in train_loader:
                            x, y = x.to(device), y.to(device)
                            output = model(x)
                            loss = loss_func1(output, y) 
                            preds = output.argmax(dim=1)
                            correct = (preds == y)
                            y_unique = torch.unique(y)
                            for i in range(len(y_unique)):
                                train_accuracy_label[y_unique[i]] += correct[y == y_unique[i]].sum().item()
                                train_num_label[y_unique[i]] += (y==y_unique[i]).sum().item()
                            acc = correct.sum().item() / x.size(0)

                            train_loss.update(loss.item(), x.size(0))
                            train_accuracy.update(acc, x.size(0))
                    for x, y in test_loader:
                            x, y = x.to(device), y.to(device)
                            output = model(x)
                            loss = loss_func1(output, y) 
                            preds = output.argmax(dim=1)
                            correct = (preds == y)
                            y_unique = torch.unique(y)
                            for i in range(len(y_unique)):
                                test_accuracy_label[y_unique[i]] += correct[y == y_unique[i]].sum().item()
                                test_num_label[y_unique[i]] += (y==y_unique[i]).sum().item()

                            acc = correct.sum().item() / x.size(0)

                            test_loss.update(loss.item(), x.size(0))
                            test_accuracy.update(acc, x.size(0))
                    train_accuracy_label /= train_num_label
                    test_accuracy_label /= test_num_label
                logging.info(f"Epoch {epoch+1}: Train Loss: {losses.avg:.4f} | Test Loss: {test_loss.avg:.4f} | Acc@1: {test_accuracy.avg:.4f}  | acc_label_train: {train_accuracy_label} | acc_label_test: {test_accuracy_label}")
                if args.epoch-epoch <=10:
                    train_loss_list.append(train_loss.avg)
                    test_loss_list.append(test_loss.avg)
                    train_acc_list.append(train_accuracy.avg)
                    test_acc_list.append(test_accuracy.avg)
                    train_acc_label_list.append(train_accuracy_label)
                    test_acc_label_list.append(test_accuracy_label)
    logging.info(f"test_acc: {np.mean(test_acc_list)}")

    train_acc_label_array = torch.stack(train_acc_label_list)
    test_acc_label_array = torch.stack(test_acc_label_list)

    train_acc_label_mean = train_acc_label_array.mean(dim=0)
    test_acc_label_mean = test_acc_label_array.mean(dim=0)


    return {
        'train_loss': np.mean(train_loss_list),
        'test_loss': np.mean(test_loss_list),
        'train_acc': np.mean(train_acc_list),
        'test_acc': np.mean(test_acc_list),
        'train_acc_per_class': train_acc_label_mean.cpu().numpy(),
        'test_acc_per_class': test_acc_label_mean.cpu().numpy(),
    }

def exp(args):
    if args.use_wandb:
        if args.wandb_run_name is None:
            timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
            args.wandb_run_name = (f"{args.dataset}_{args.mechanism}_"
                                f"eps{args.epsilon}_delta{args.delta}_sigma{args.sigma}_"
                                f"Time{timestamp}")
        # Set offline mode if specified
        if args.wandb_offline:
            os.environ['WANDB_MODE'] = 'offline'

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                'task': args.task,
                'dataset': args.dataset,
                'mechanism': args.mechanism,
                'epsilon': args.epsilon,
                'epsilon_p': args.epsilon_p,
                'sigma': args.sigma,
                'delta': args.delta,
                'alpha': args.alpha,
                'beta': args.beta,
                'alpha1': args.alpha1,
                'lr': args.lr,
                'momentum': args.momentum,
                'weight_decay': args.weight_decay,
                'epoch': args.epoch,
                'batch_size': args.batch_size,
                'kfolds': args.kfolds,
                'num_exp': args.num_exp,
                'arch': args.arch,
            },
            group=f"{args.dataset}_{args.mechanism}_eps{args.epsilon}",
        )

    seed_list = np.random.randint(0, 2025, args.num_exp).tolist()

    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []
    train_acc_per_class_list = []
    test_acc_per_class_list = []

    for i, seed in enumerate(tqdm(seed_list, desc="Running experiments")):
        setattr(args, 'seed', seed)
        logging.info(f"\n--- Experiment {i+1}/{args.num_exp} (seed={seed}) ---")
        
        results = run_single_experiment(args)

        train_loss_list.append(results['train_loss'])
        test_loss_list.append(results['test_loss'])
        train_acc_list.append(results['train_acc'])
        test_acc_list.append(results['test_acc'])
        train_acc_per_class_list.append(results['train_acc_per_class'])
        test_acc_per_class_list.append(results['test_acc_per_class'])



    results_summary = {
        # Loss Statistics
        'train_loss_mean': np.mean(train_loss_list),
        'train_loss_std': np.std(train_loss_list),
        'train_loss_list': train_loss_list,
        
        'test_loss_mean': np.mean(test_loss_list),
        'test_loss_std': np.std(test_loss_list),
        'test_loss_list': test_loss_list,
        
        # Accuracy
        'train_acc_mean': np.mean(train_acc_list),
        'train_acc_std': np.std(train_acc_list),
        'train_acc_list': train_acc_list,
        
        'test_acc_mean': np.mean(test_acc_list),
        'test_acc_std': np.std(test_acc_list),
        'test_acc_list': test_acc_list,
        
        # Per-class accuracy
        'train_acc_per_class_mean': np.mean(train_acc_per_class_list, axis=0),
        'test_acc_per_class_mean': np.mean(test_acc_per_class_list, axis=0),
        'train_acc_per_class_std': np.std(train_acc_per_class_list, axis=0),
        'test_acc_per_class_std': np.std(test_acc_per_class_list, axis=0),

    }

    logging.info(f"Number of experiments: {args.num_exp}")
    logging.info(f"Seeds: {seed_list}")
    logging.info(f"{'-'*80}")
    logging.info(f"Train loss list: {results_summary['train_loss_list']}")
    logging.info(f"Test loss list: {results_summary['test_loss_list']}")
    logging.info(f"Train acc list: {results_summary['train_acc_list']}")
    logging.info(f"Test loss list: {results_summary['test_acc_list']}")
    logging.info(f"{'-'*80}")
    logging.info("LOSS RESULTS:")
    logging.info(f"{'-'*80}")
    logging.info(f"Train Loss: {results_summary['train_loss_mean']:.4f} ± {results_summary['train_loss_std']:.4f}")
    logging.info(f"Test Loss:  {results_summary['test_loss_mean']:.4f} ± {results_summary['test_loss_std']:.4f}")
    logging.info(f"{'-'*80}")
    logging.info("ACCURACY RESULTS:")
    logging.info(f"{'-'*80}")
    logging.info(f"Train Acc:  {results_summary['train_acc_mean']:.4f} ± {results_summary['train_acc_std']:.4f}")
    logging.info(f"Test Acc:   {results_summary['test_acc_mean']:.4f} ± {results_summary['test_acc_std']:.4f}")
    logging.info(f"{'-'*80}")
    logging.info("PER-CLASS ACCURACY:")
    logging.info(f"{'-'*80}")
    logging.info(f"Train (per class): {results_summary['train_acc_per_class_mean']}")
    logging.info(f"Test (per class):  {results_summary['test_acc_per_class_mean']}")
    logging.info(f"Train (per class) Std: {results_summary['train_acc_per_class_std']}")
    logging.info(f"Test (per class) Std: {results_summary['test_acc_per_class_std']}")
    logging.info( "="*80)

    logging.info(f"macro-P: {results_summary['macro_P_mean']}")
    logging.info(f"macro-P-std: {results_summary['macro_P_std']}")
    logging.info(f"macro-R: {results_summary['macro_R_mean']}")
    logging.info(f"macro-R-std: {results_summary['macro_R_std']}")
    logging.info(f"macro-F1: {results_summary['macro_F1_mean']}")
    logging.info(f"macro-F1-std: {results_summary['macro_F1_std']}")

    # Log final summary to wandb
    if args.use_wandb:
        wandb.log({
            'train_loss_list': results_summary['train_loss_list'],
            'test_loss_list': results_summary['test_loss_list'],
            'train_acc_list': results_summary['train_acc_list'],
            'test_acc_list': results_summary['test_acc_list'],
            'train_loss_mean': results_summary['train_loss_mean'],
            'train_loss_std': results_summary['train_loss_std'],
            'test_loss_mean': results_summary['test_loss_mean'],
            'test_loss_std': results_summary['test_loss_std'],
            'train_acc_mean': results_summary['train_acc_mean'],
            'train_acc_std': results_summary['train_acc_std'],
            'test_acc_mean': results_summary['test_acc_mean'],
            'test_acc_std': results_summary['test_acc_std'],
            'train_acc_per_class_mean': results_summary['train_acc_per_class_mean'],
            'test_acc_per_class_mean': results_summary['test_acc_per_class_mean'],
            'train_acc_per_class_std': results_summary['train_acc_per_class_std'],
            'test_acc_per_class_std': results_summary['test_acc_per_class_std'],
        })
        # Create summary table
        summary_table = wandb.Table(
            columns=['Metric', 'Mean', 'Std'],
            data=[
                ['Train Loss', results_summary['train_loss_mean'], results_summary['train_loss_std']],
                ['Test Loss', results_summary['test_loss_mean'], results_summary['test_loss_std']],
                ['Train Acc', results_summary['train_acc_mean'], results_summary['train_acc_std']],
                ['Test Acc', results_summary['test_acc_mean'], results_summary['test_acc_std']],
            ]
        )
        wandb.log({'summary_table': summary_table})
        
        wandb.finish()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification Training with LabelDP")

    parser.add_argument('--task', type=str, default='classification', choices=['classification'], help='Task type')
    parser.add_argument('--dataset', type=str, default='cifar10_1', choices=['cifar10_1', 'cifar10_2', 'cifar10_3'], help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='../data', help='Dataset directory')
    parser.add_argument('--arch', type=str, default='resnet18v2', help='Model architecture name')
    parser.add_argument('--mechanism', type=str, default='rrwithweight', choices=['rr', 'rrwithweight', 'rrwithprior'], help='LabelDP mechanism')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Privacy budget ε for LabelDP')
    parser.add_argument('--epsilon_p', type=float, default=1.0, help='Privacy budget for distribution estimation')
    parser.add_argument('--sigma', type=float, default=1.0, help='Standard deviation of weight matrix')
    parser.add_argument('--delta', type=int, default=1, help='Topk of diagonal of weight matrix')
    parser.add_argument('--prior', type=str, default=None, help='Label prior distribution (e.g., uniform)')
    parser.add_argument('--alpha', type=float, default=1.0, help='SCE alpha coefficient')
    parser.add_argument('--beta', type=float, default=0.0, help='SCE beta coefficient')
    parser.add_argument('--alpha1', type=float, default=8.0, help='Mixup alpha parameter')
    parser.add_argument('--lr', type=float, default=0.4, help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 regularization)')
    parser.add_argument('--epoch', type=int, default=160, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training and testing')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed')
    parser.add_argument('--log_dir', type=str, default='./logs2', help='Directory to save log files')
    parser.add_argument('--kfolds', type=float, default=0.99, help='k of kfolds')
    parser.add_argument('--num_exp', type=int, default=10, help='Number of experiment')

    # Wandb arguments
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb_offline', action='store_true', help='Run wandb in offline mode')
    parser.add_argument('--wandb_project', type=str, default='dp', help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name (auto-generated if None)')


    args = parser.parse_args()

    init_logger(args.log_dir)

    exp(args)


