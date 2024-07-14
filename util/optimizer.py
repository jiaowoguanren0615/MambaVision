import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List
import operator
import functools
from copy import copy
from math import sqrt



__all__ = ['SophiaG', 'AdaFactor', 'LAMB']

# optimizer = SophiaG(model.parameters(), lr=args.lr, betas=(0.965, 0.99), rho=0.01, weight_decay=1e-1)

# TODO https://arxiv.org/pdf/2209.06794v4
# optimizer = AdaFactor(model.parameters(), lr=5e-3, beta1=0.0, beta2=0.8)

class SophiaG(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho=0.04,
                 weight_decay=1e-1, *, maximize: bool = False,
                 capturable: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= rho:
            raise ValueError("Invalid rho parameter at index 1: {}".format(rho))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, rho=rho,
                        weight_decay=weight_decay,
                        maximize=maximize, capturable=capturable)
        super(SophiaG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('capturable', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    @torch.no_grad()
    def update_hessian(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                if 'hessian' not in state.keys():
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['hessian'].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

    @torch.no_grad()
    def step(self, closure=None, bs=5120):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            state_steps = []
            hessian = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)

                if p.grad.is_sparse:
                    raise RuntimeError('Hero does not support sparse gradients')
                grads.append(p.grad)
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                if 'hessian' not in state.keys():
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                state_steps.append(state['step'])
                hessian.append(state['hessian'])

                if self.defaults['capturable']:
                    bs = torch.ones((1,), dtype=torch.float, device=p.device) * bs

            sophiag(params_with_grad,
                    grads,
                    exp_avgs,
                    hessian,
                    state_steps,
                    bs=bs,
                    beta1=beta1,
                    beta2=beta2,
                    rho=group['rho'],
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    maximize=group['maximize'],
                    capturable=group['capturable'])

        return loss


def sophiag(params: List[Tensor],
            grads: List[Tensor],
            exp_avgs: List[Tensor],
            hessian: List[Tensor],
            state_steps: List[Tensor],
            capturable: bool = False,
            *,
            bs: int,
            beta1: float,
            beta2: float,
            rho: float,
            lr: float,
            weight_decay: float,
            maximize: bool):
    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    func = _single_tensor_sophiag

    func(params,
         grads,
         exp_avgs,
         hessian,
         state_steps,
         bs=bs,
         beta1=beta1,
         beta2=beta2,
         rho=rho,
         lr=lr,
         weight_decay=weight_decay,
         maximize=maximize,
         capturable=capturable)


def _single_tensor_sophiag(params: List[Tensor],
                           grads: List[Tensor],
                           exp_avgs: List[Tensor],
                           hessian: List[Tensor],
                           state_steps: List[Tensor],
                           *,
                           bs: int,
                           beta1: float,
                           beta2: float,
                           rho: float,
                           lr: float,
                           weight_decay: float,
                           maximize: bool,
                           capturable: bool):
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        hess = hessian[i]
        step_t = state_steps[i]

        if capturable:
            assert param.is_cuda and step_t.is_cuda and bs.is_cuda

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            hess = torch.view_as_real(hess)
            param = torch.view_as_real(param)

        # update step
        step_t += 1

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

        if capturable:
            step = step_t
            step_size = lr
            step_size_neg = step_size.neg()

            ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None, 1)
            param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)
        else:
            step = step_t.item()
            step_size_neg = - lr

            ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None, 1)
            param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)



class AdaFactor(Optimizer):
    def __init__(self, params, lr=None, beta1=0.9, beta2=0.999, eps1=1e-30,
                 eps2=1e-3, cliping_threshold=1, non_constant_decay=True,
                 enable_factorization=True, ams_grad=True, weight_decay=0):

        enable_momentum = beta1 != 0
        self.beta1_glob = copy(beta1)
        self.beta2_glob = copy(beta2)
        self.lr_glob = copy(lr)

        beta1 = self.beta1_glob if hasattr(beta1, '__call__') else lambda x: self.beta1_glob
        beta2 = self.beta2_glob if hasattr(beta2, '__call__') else lambda x: self.beta2_glob

        if non_constant_decay:
            ams_grad = False
            if isinstance(self.beta1_glob, float):
                beta1 = lambda t: self.beta1_glob * (1 - self.beta1_glob ** (t - 1)) / (1 - self.beta1_glob ** t)
            if isinstance(self.beta2_glob, float):
                beta2 = lambda t: self.beta2_glob * (1 - self.beta2_glob ** (t - 1)) / (1 - self.beta2_glob ** t)

        relative_step_size = True

        if lr is None:
            # default value from article
            lr = lambda t: min(1e-2, 1 / sqrt(t))

        if isinstance(self.lr_glob, float):
            lr = lambda x: self.lr_glob
            relative_step_size = False

        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps1=eps1,
                        eps2=eps2, cliping_threshold=cliping_threshold, weight_decay=weight_decay, ams_grad=ams_grad,
                        enable_factorization=enable_factorization,
                        enable_momentum=enable_momentum, relative_step_size=relative_step_size)

        super(AdaFactor, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdaFactor, self).__setstate__(state)

    def _experimental_reshape(self, shape):
        temp_shape = shape[2:]
        if len(temp_shape) == 1:
            new_shape = (shape[0], shape[1] * shape[2])
        else:
            tmp_div = len(temp_shape) // 2 + len(temp_shape) % 2
            new_shape = (shape[0] * functools.reduce(operator.mul, temp_shape[tmp_div:], 1),
                         shape[1] * functools.reduce(operator.mul, temp_shape[:tmp_div], 1))
        return new_shape, copy(shape)

    def _check_shape(self, shape):
        '''
        output1 - True - algorithm for matrix, False - vector;
        output2 - need reshape
        '''
        if len(shape) > 2:
            return True, True
        elif len(shape) == 2:
            return True, False
        elif len(shape) == 2 and (shape[0] == 1 or shape[1] == 1):
            return False, False
        else:
            return False, False

    def _rms(self, x):
        return sqrt(torch.mean(x.pow(2)))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                data_backup = p.data.clone().detach()

                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                is_matrix, is_need_reshape = self._check_shape(grad.size())
                new_shape = p.data.size()
                if is_need_reshape and group['enable_factorization']:
                    new_shape, old_shape = \
                        self._experimental_reshape(p.data.size())
                    grad = grad.view(new_shape)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    if group['enable_momentum']:
                        state['exp_avg'] = torch.zeros(new_shape, dtype=torch.float32, device=p.grad.device)

                    if is_matrix and group['enable_factorization']:
                        state['exp_avg_sq_R'] = torch.zeros((1, new_shape[1]), dtype=torch.float32,
                                                            device=p.grad.device)
                        state['exp_avg_sq_C'] = torch.zeros((new_shape[0], 1), dtype=torch.float32,
                                                            device=p.grad.device)
                    else:
                        state['exp_avg_sq'] = torch.zeros(new_shape, dtype=torch.float32, device=p.grad.device)
                    if group['ams_grad']:
                        state['exp_avg_sq_hat'] = torch.zeros(new_shape, dtype=torch.float32, device=p.grad.device)

                if group['enable_momentum']:
                    exp_avg = state['exp_avg']

                if is_matrix and group['enable_factorization']:
                    exp_avg_sq_R = state['exp_avg_sq_R']
                    exp_avg_sq_C = state['exp_avg_sq_C']
                else:
                    exp_avg_sq = state['exp_avg_sq']

                if group['ams_grad']:
                    exp_avg_sq_hat = state['exp_avg_sq_hat']

                state['step'] += 1
                lr_t = group['lr'](state['step'])
                if group['relative_step_size']:
                    lr_t *= max(group['eps2'], self._rms(p.data))

                if group['enable_momentum']:
                    beta1_t = group['beta1'](state['step'])
                    exp_avg.mul_(beta1_t).add_(1 - beta1_t, grad)

                beta2_t = group['beta2'](state['step'])

                if is_matrix and group['enable_factorization']:
                    exp_avg_sq_R.mul_(beta2_t).add_(1 - beta2_t,
                                                    torch.sum(torch.mul(grad, grad).add_(group['eps1']), dim=0,
                                                              keepdim=True))
                    exp_avg_sq_C.mul_(beta2_t).add_(1 - beta2_t,
                                                    torch.sum(torch.mul(grad, grad).add_(group['eps1']), dim=1,
                                                              keepdim=True))
                    v = torch.mul(exp_avg_sq_C, exp_avg_sq_R).div_(torch.sum(exp_avg_sq_R))
                else:
                    exp_avg_sq.mul_(beta2_t).addcmul_(1 - beta2_t, grad, grad).add_((1 - beta2_t) * group['eps1'])
                    v = exp_avg_sq

                g = grad
                if group['enable_momentum']:
                    g = torch.div(exp_avg, 1 - beta1_t ** state['step'])

                if group['ams_grad']:
                    torch.max(exp_avg_sq_hat, v, out=exp_avg_sq_hat)
                    v = exp_avg_sq_hat
                    u = torch.div(g, (torch.div(v, 1 - beta2_t ** state['step'])).sqrt().add_(group['eps1']))
                else:
                    u = torch.div(g, v.sqrt())

                u.div_(max(1, self._rms(u) / group['cliping_threshold']))
                p.data.add_(-lr_t * (u.view(old_shape) if is_need_reshape and group['enable_factorization'] else u))

                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'] * lr_t, data_backup)

        return loss



# Define the LAMB optimizer
# See paper [Large Batch Optimization for Deep Learning: Training BERT in 76 minutes](https://arxiv.org/abs/1904.00962).
class LAMB(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01):
        if not lr > 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(LAMB, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('LAMB does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = (exp_avg_sq.sqrt() / torch.sqrt(torch.tensor(bias_correction2))).add_(group['eps'])
                step_size = group['lr'] / bias_correction1

                # LAMB update
                weight_norm = torch.norm(p.data)
                update = exp_avg / denom
                update_norm = torch.norm(update)
                if weight_norm == 0 or update_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / update_norm

                p.data.add_(-step_size * trust_ratio, update)

                # Weight decay
                if group['weight_decay'] != 0:
                    p.data.add_(-group['lr'] * group['weight_decay'], p.data)

        return loss