import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--noise_scale', type=float, default=0.0)
parser.add_argument('--point_type', type=str, choices=['uniform', 'saddle', 'center', 'spiral'], default='spiral')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

if args.point_type == 'spiral':
    true_y0s = [
        torch.tensor([[0.1, 0.1]]).to(device),
        torch.tensor([[0.1, -0.1]]).to(device)
    ]
    t = torch.linspace(0, 7., args.data_size).to(device)
    true_A = torch.tensor([[0.3, -1.], [1., 0.3]]).to(device)
    test_true_y0 = torch.tensor([[-0.2, 0.1]]).to(device)
elif args.point_type == 'saddle':
    true_y0s = [
        torch.tensor([[2.0, 0.1]]).to(device),
        torch.tensor([[-2.0, 0.15]]).to(device)
    ]
    t = torch.linspace(0, 3., args.data_size).to(device)
    true_A = torch.tensor([[-1., 0.], [0., 1.]]).to(device)
    test_true_y0 = torch.tensor([[-2.0, -0.2]]).to(device)
elif args.point_type == 'center':
    true_y0s = [
        torch.tensor([[0.5, 0.4]]).to(device),
        torch.tensor([[-0.7, -1.3]]).to(device)
    ]
    t = torch.linspace(0, 10., args.data_size).to(device)
    true_A = torch.tensor([[0., -1], [1., 0.]]).to(device)
    test_true_y0 = torch.tensor([[1.1, -0.5]]).to(device)
elif args.point_type == 'uniform':
    true_y0s = [
        torch.tensor([[0.1, 0.3]]).to(device),
        torch.tensor([[0.2, -0.2]]).to(device),
    ]
    t = torch.linspace(0., 3., args.data_size).to(device)
    true_A = torch.tensor([[1., 0.], [0., 1.]]).to(device)
    test_true_y0 = torch.tensor([[-0.1, 0.2]]).to(device)

test_t = t.clone().detach()

class Lambda(nn.Module):

    def forward(self, t, y):
        return torch.mm(y, true_A)


with torch.no_grad():
    true_ys = [odeint(Lambda(), true_y0, t, method='dopri5') for true_y0 in true_y0s]
    test_true_y = odeint(Lambda(), test_true_y0, test_t, method='dopri5')


def get_batch():
    traj_idx = np.random.randint(len(true_y0s))
    true_y0 = true_y0s[traj_idx]
    true_y = true_ys[traj_idx]
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    noise_y = torch.randn_like(batch_y) * args.noise_scale
    batch_y += noise_y
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs(f'png/{args.point_type}/{args.noise_scale}')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 10), facecolor='white')
    ax_traj = fig.add_subplot(221, frameon=False)
    ax_phase = fig.add_subplot(222, frameon=False)
    ax_vecfield = fig.add_subplot(223, frameon=False)
    ax_loss = fig.add_subplot(224, frameon=False)
    plt.show(block=False)


def visualize(true_ys, pred_ys, test_true_y, test_pred_y, odefunc, itr, loss_list, test_loss_list):

    if args.viz:
        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x')
        for true_y, pred_y in zip(true_ys, pred_ys):
            ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], '--', color='blue', label='train true')
            ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '-', color='blue', label='train pred')
        ax_traj.plot(test_t.cpu().numpy(), test_true_y.cpu().numpy()[:, 0, 0], '--', color='red', label='test true')
        ax_traj.plot(test_t.cpu().numpy(), test_pred_y.cpu().numpy()[:, 0, 0],'-', color='red', label='test pred')
        ax_traj.set_xlim(min(t.cpu().min(), test_t.cpu().min()), max(t.cpu().max(), test_t.cpu().max()))
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        for true_y, pred_y in zip(true_ys, pred_ys):
            ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], '--', color='blue', label='train true')
            ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], '-', color='blue', label='train pred')
        ax_phase.plot(test_true_y.cpu().numpy()[:, 0, 0], test_true_y.cpu().numpy()[:, 0, 1], '--', color='red', label='test true')
        ax_phase.plot(test_pred_y.cpu().numpy()[:, 0, 0], test_pred_y.cpu().numpy()[:, 0, 1], '-', color='red', label='test pred')
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)
        ax_phase.legend()

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        true_dydt = torch.mm(torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device), true_A.T).cpu().detach().numpy()
        true_mag = np.sqrt(true_dydt[:, 0]**2 + true_dydt[:, 1]**2).reshape(-1, 1)
        true_dydt = (true_dydt / true_mag)
        true_dydt = true_dydt.reshape(21, 21, 2)

        learned_dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
        learned_mag = np.sqrt(learned_dydt[:, 0]**2 + learned_dydt[:, 1]**2).reshape(-1, 1)
        learned_dydt = (learned_dydt / learned_mag)
        learned_dydt = learned_dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, true_dydt[:, :, 0], true_dydt[:, :, 1], color="gray", density=[0.5, 1], linewidth=1)
        ax_vecfield.streamplot(x, y, learned_dydt[:, :, 0], learned_dydt[:, :, 1], color="black", density=[0.5, 1], linewidth=1)
        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)

        ax_loss.cla()
        ax_loss.set_title('Loss')
        ax_loss.semilogy(loss_list, label='Train', color='blue')
        ax_loss.semilogy(test_loss_list, label='Test', color='red')
        ax_loss.set_xlim(0, args.niters)
        ax_loss.set_ylim(0, 1.1 * max(test_loss_list))
        ax_loss.set_xlabel('Iteration')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()

        fig.tight_layout()
        plt.savefig(f'png/{args.point_type}/{args.noise_scale}/{itr:03d}')


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':

    ii = 0

    func = ODEFunc().to(device)
    
    optimizer = optim.AdamW(func.parameters(), lr=1e-3)
    criterion = nn.L1Loss()
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)
    test_loss_meter = RunningAverageMeter(0.97)

    loss_list = []
    test_loss_list = []

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(func, batch_y0, batch_t).to(device)
        loss = criterion(pred_y, batch_y)
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())
        loss_list.append(loss_meter.avg)

        with torch.no_grad():
            test_pred_y = odeint(func, test_true_y0, test_t)
            test_loss = torch.mean(torch.abs(test_pred_y - test_true_y))
            test_loss_meter.update(test_loss.item())
            test_loss_list.append(test_loss_meter.avg)

            if itr % args.test_freq == 0:
                pred_ys = [odeint(func, true_y0, t) for true_y0 in true_y0s]
                loss = np.mean([criterion(pred_y, true_y).item() for pred_y, true_y in zip(pred_ys, true_ys)])
                
                print(f'Iter {itr:04d} | Train Loss {loss:.6f} | Test Loss {test_loss.item():.6f}')
                visualize(true_ys, pred_ys, test_true_y, test_pred_y, func, ii, loss_list, test_loss_list)
                ii += 1

        end = time.time()