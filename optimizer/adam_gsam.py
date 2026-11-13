from typing import List

import torch
from torch import Tensor

from utility.args import Args
from utility.loss import smooth_crossentropy

from .adam_sam import _adam_l2_step


class Adam_GSAM(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr: float = 1e-3,
            betas=(0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 1e-2,
            maximize: bool = False,
            rho: float = 0.3,
            alpha: float = 0.01,
            adaptive: bool = False,
            perturb_eps: float = 1e-12,
            ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
            rho=rho,
            alpha=alpha,
            adaptive=adaptive,
            perturb_eps=perturb_eps,
        )
        super(Adam_GSAM, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(p).detach()
                    if "exp_avg_sq" not in state:
                        state["exp_avg_sq"] = torch.zeros_like(p).detach()

    def _grad_norm(self, by=None, weight_adaptive: bool = False):
        if by is None:
            return torch.norm(
                torch.stack([
                    (((torch.abs(p) if weight_adaptive else 1.0) * p.grad).norm(p=2))
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]),
                p=2,
            )
        return torch.norm(
            torch.stack([
                (((torch.abs(p) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2))
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )

    @torch.no_grad()
    def perturb_weights(self):
        grad_norm = self._grad_norm(weight_adaptive=self.param_groups[0]["adaptive"])
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + group["perturb_eps"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                self.state[p]["old_g"] = p.grad.detach().clone()
                e_w = p.grad * scale.to(p)
                if group["adaptive"]:
                    e_w = e_w * torch.pow(p, 2)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def unperturb(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if "old_p" in self.state[p]:
                    p.data.copy_(self.state[p]["old_p"])

    @torch.no_grad()
    def gradient_decompose(self):
        alpha = self.param_groups[0]["alpha"]
        eps = self.param_groups[0]["perturb_eps"]
        inner_prod = torch.tensor(0.0, device=self.param_groups[0]["params"][0].device)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                inner_prod = inner_prod + torch.sum(self.state[p]["old_g"] * p.grad)

        new_grad_norm = self._grad_norm()
        old_grad_norm = self._grad_norm(by="old_g")
        cosine = inner_prod / (new_grad_norm * old_grad_norm + eps)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                vertical = self.state[p]["old_g"] - cosine * old_grad_norm * p.grad / (new_grad_norm + eps)
                p.grad.add_(vertical, alpha=-alpha)

    def step(self, model, inputs, targets):
        predictions = model(inputs)
        loss = smooth_crossentropy(predictions, targets)
        loss.mean().backward()

        self.perturb_weights()
        self.zero_grad()

        model.module.setBatchNormTracking(False)
        loss_intermediate = smooth_crossentropy(model(inputs), targets)
        model.module.setBatchNormTracking(True)
        loss_intermediate.mean().backward()

        self.gradient_decompose()
        self.unperturb()

        if Args.grad_clip_norm != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Args.grad_clip_norm)

        self.inner_step()
        self.zero_grad()
        return loss, predictions

    @torch.no_grad()
    def inner_step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            exp_avgs: List[Tensor] = []
            exp_avg_sqs: List[Tensor] = []
            state_steps: List[int] = []
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Adam_GSAM does not support sparse gradients")
                params_with_grad.append(p)
                grads.append(p.grad)
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                state["step"] += 1
                state_steps.append(state["step"])

            _adam_l2_step(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
            )

        return loss
