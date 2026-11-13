import torch

from utility.args import Args
from utility.loss import smooth_crossentropy

from .adamW_sam import adam
from .orthosam import _ns_project, _should_use_orthosam


class AdamW_OrthoSAM(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr: float = 1e-3,
            betas=(0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 1e-2,
            maximize: bool = False,
            rho: float = 0.01,
            ns_steps: int = 5,
            grad_eps: float = 1e-12,
            ):
        parameter_entries = list(params)
        if len(parameter_entries) == 0:
            raise ValueError("AdamW_OrthoSAM received an empty parameter list.")

        named_entries = []
        for entry in parameter_entries:
            if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], str):
                named_entries.append((entry[0], entry[1]))
            else:
                named_entries.append(("", entry))

        parameters = [param for _, param in named_entries]
        parameter_name_map = {id(param): name for name, param in named_entries}

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
            rho=rho,
            ns_steps=ns_steps,
            grad_eps=grad_eps,
        )
        super(AdamW_OrthoSAM, self).__init__(parameters, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    state["param_name"] = parameter_name_map.get(id(p), "")
                    state["use_ns"] = _should_use_orthosam(p, state["param_name"])
                    if "step" not in state:
                        state["step"] = 0
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(p).detach()
                    if "exp_avg_sq" not in state:
                        state["exp_avg_sq"] = torch.zeros_like(p).detach()

    def _grad_norm(self):
        grads = [
            p.grad.norm(p=2)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]

        if not grads:
            first_param = self.param_groups[0]["params"][0]
            return torch.zeros((), device=first_param.device)

        return torch.norm(torch.stack(grads), p=2)

    @torch.no_grad()
    def move_up(self):
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            rho = group["rho"]
            denom = grad_norm + group["grad_eps"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if self.state[p]["use_ns"]:
                    direction = _ns_project(p.grad, steps=ns_steps, apply_matrix_scale=False)
                    direction = direction / (direction.norm() + group["grad_eps"])
                else:
                    direction = p.grad / denom

                e_w = rho * direction
                p.add_(e_w)
                self.state[p]["e_w"] = e_w.detach().clone()

    @torch.no_grad()
    def move_back(self):
        for group in self.param_groups:
            for p in group["params"]:
                if "e_w" in self.state[p]:
                    p.sub_(self.state[p]["e_w"])
                    del self.state[p]["e_w"]

    def step(self, model, inputs, targets):
        predictions = model(inputs)

        loss = smooth_crossentropy(predictions, targets)
        loss.mean().backward()

        self.move_up()
        self.zero_grad()

        model.module.setBatchNormTracking(False)
        loss_intermediate = smooth_crossentropy(model(inputs), targets)
        model.module.setBatchNormTracking(True)
        loss_intermediate.mean().backward()

        self.move_back()

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
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("AdamW_OrthoSAM does not support sparse gradients")

                params_with_grad.append(p)
                state = self.state[p]
                if state["use_ns"]:
                    grads.append(_ns_project(p.grad, steps=group["ns_steps"], apply_matrix_scale=False))
                else:
                    grads.append(p.grad)

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                state["step"] += 1
                state_steps.append(state["step"])

            adam(params_with_grad,
                 grads,
                 exp_avgs,
                 exp_avg_sqs,
                 max_exp_avg_sqs,
                 state_steps,
                 beta1=beta1,
                 beta2=beta2,
                 lr=group["lr"],
                 weight_decay=group["weight_decay"],
                 eps=group["eps"],
                 maximize=False)

        return loss
