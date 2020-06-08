import torch
#from torch.optimizer import Optimizer, required
from torch.optim import Optimizer
import torch.distributed as dist


class SGDOPO_W(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v_{t+1} = \mu * v_{t} + g_{t+1} \\
                  p_{t+1} = p_{t} - lr * v_{t+1}

        where p, g, v and :math:`\mu` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v_{t+1} = \mu * v_{t} + lr * g_{t+1} \\
             p_{t+1} = p_{t} - v_{t+1}

        The Nesterov version is analogously modified.
    """

    def __init__(self, model, lr=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, alpha=1.0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDOPO_W, self).__init__(model.parameters(), defaults)

        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov

        self.alpha = alpha
        self.param_to_handle = {}
        self.model = model
        self.prefix = "opt"
        self.cnt = 0
        print("SGDOPO_W alpha %f", self.alpha)

        #register buffer and hook
        self.register_buffer_and_hook()

    def __setstate__(self, state):
        super(SGDOPO_W, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def register_buffer_and_hook(self):
        module_list = []
        # gather all modules
        def gather_all_module(submodule):
            has_child = False
            for m in submodule.children():
                has_child = True
                gather_all_module(m)
            if not has_child:
                module_list.append(submodule)
        
        gather_all_module(self.model)

        def wait_update_reduce_hook(m, gi, go):
            for key, param in m.named_parameters():
                if param.data is not None and param.requires_grad:
                    if param.grad is None :
                        #print(key)
                        continue
                    grad = param.grad.data
                    if self.weight_decay != 0:
                        grad.add_(self.weight_decay, param.data)
                    if self.momentum != 0:
                        param_state = self.state[param]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = grad.clone().detach().requires_grad_(False) #torch.clone(grad).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(self.momentum).add_(1 - self.dampening, grad)
                        if self.nesterov:
                            grad = grad.add(self.momentum, buf)
                        else:
                            grad = buf
                    
                    #wait for reduce
                    red = self.param_to_handle.get(param)
                    if red is not None :
                        if not red.is_completed():
                            red.wait()
                    ave_param = m._buffers[self.prefix+key]
                    if self.alpha != 1.0:
                        ave_param.mul_(self.alpha).add_(-self.lr, grad)
                    else:
                        ave_param.add_(-self.lr, grad)
                    param.data.copy_(ave_param)

                    #clear grad and detach
                    param.grad.detach_()
                    param.grad.zero_()

                    #begin reduce for next iteration
                    #del self.param_to_handle[param]
                    self.param_to_handle[param] = dist.all_reduce(ave_param, dist.ReduceOp.SUM, async_op=True)


                    
                    


        for m in module_list:
            need_hook = False
            for key, param in m.named_parameters():
                #print("prm===>", key)
                if param.data is not None and param.requires_grad:
                    #add reduce buffer
                    m.register_buffer(self.prefix+key, param.data.clone().detach().requires_grad_(False))
                    need_hook = True
            if need_hook:
                m.register_backward_hook(wait_update_reduce_hook)
                
        #print(module_list)
        print("modules len ", len(module_list))

    def step(self, closure=None):
        print("calling step")
        rank = dist.get_rank()
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                if self.alpha==1.0:
                    p.data.add_(-group['lr'], d_p)
                else:
                    print("error")
                    #tmpbuf = self.kv[p]
                    #tmpbuf.mul_(self.alpha).add_(-group['lr'], d_p)
                    #p.data.copy_(tmpbuf)
        #if rank == 0 and self.cnt >= 102:
        #    print("%d sgd aw reduce weight" % (self.cnt))
        #if rank == 0 and self.cnt < 102:
        #    print("%d sgd aw not reduce weight" % (self.cnt))
        #self.cnt += 1
        return loss
