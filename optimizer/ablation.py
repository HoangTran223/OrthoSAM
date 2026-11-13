import torch
import torch.nn.functional as F

from utility.args import Args
from utility.loss import smooth_crossentropy

from .orthosam import _ns_project, _should_use_orthosam

PERTURB_NONE = "none"
PERTURB_GRAD = "grad"
PERTURB_ORTHO = "ortho"
_PERTURBATIONS = (PERTURB_NONE, PERTURB_GRAD, PERTURB_ORTHO)

DESCENT_SGD = "sgd"
DESCENT_ORTHO = "ortho"
DESCENT_MUON_CANONICAL = "muon_canonical"
_DESCENTS = (DESCENT_SGD, DESCENT_ORTHO, DESCENT_MUON_CANONICAL)


class AblationFamily(torch.optim.Optimizer):

    def __init__(
            self,
            params,
            lr: float = 1e-1,
            momentum: float = 0.9,
            weight_decay: float = 1e-2,
            nesterov: bool = False,
            rho: float = 0.1,
            perturbation: str = PERTURB_GRAD,
            descent: str = DESCENT_SGD,
            ns_steps: int = 5,
            eps: float = 1e-12,
            ):
        if perturbation not in _PERTURBATIONS:
            raise ValueError(f"Unknown perturbation '{perturbation}'. Expected one of {_PERTURBATIONS}.")
        if descent not in _DESCENTS:
            raise ValueError(f"Unknown descent '{descent}'. Expected one of {_DESCENTS}.")

        parameter_entries = list(params)
        if len(parameter_entries) == 0:
            raise ValueError("AblationFamily received an empty parameter list.")

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
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            rho=rho,
            ns_steps=ns_steps,
            eps=eps,
        )
        super(AblationFamily, self).__init__(parameters, defaults)

        self.perturbation = perturbation
        self.descent = descent

        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    state["param_name"] = parameter_name_map.get(id(p), "")
                    state["use_ns"] = _should_use_orthosam(p, state["param_name"])
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p).detach()

    @property
    def uses_ascent_step(self) -> bool:
        return self.perturbation != PERTURB_NONE

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
            denom = grad_norm + group["eps"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if self.perturbation == PERTURB_ORTHO and self.state[p]["use_ns"]:
                    direction = _ns_project(p.grad, steps=ns_steps, apply_matrix_scale=False)
                    direction = direction / (direction.norm() + group["eps"])
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

    def step(self, model=None, inputs=None, targets=None, regression: bool = False, closure=None):
        if model is None:
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            self._descent_step()
            return loss

        predictions = model(inputs)
        loss = self._loss(predictions, targets, regression)
        loss.backward()

        if self.uses_ascent_step:
            self.move_up()
            self.zero_grad()

            model.module.setBatchNormTracking(False)
            perturbed_predictions = model(inputs)
            perturbed_loss = self._loss(perturbed_predictions, targets, regression)
            model.module.setBatchNormTracking(True)
            perturbed_loss.backward()

            self.move_back()

        if Args.grad_clip_norm != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Args.grad_clip_norm)
        self._descent_step()
        self.zero_grad()

        return loss, predictions

    @staticmethod
    def _loss(predictions, targets, regression: bool):
        if regression:
            return F.mse_loss(predictions.squeeze(-1), targets.float())
        return smooth_crossentropy(predictions, targets).mean()

    @torch.no_grad()
    def _descent_step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                use_ns = state["use_ns"]

                if self.descent == DESCENT_MUON_CANONICAL and use_ns:
                    self._canonical_muon_update(p, state, lr, momentum, weight_decay, nesterov, ns_steps)
                    continue

                if self.descent == DESCENT_ORTHO and use_ns:
                    update = _ns_project(p.grad, steps=ns_steps, apply_matrix_scale=False)
                else:
                    update = p.grad

                if weight_decay != 0:
                    update = update.add(p, alpha=weight_decay)

                if momentum != 0:
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(update)
                    update = update.add(buf, alpha=momentum) if nesterov else buf

                p.add_(update, alpha=-lr)

    @torch.no_grad()
    def _canonical_muon_update(self, p, state, lr, momentum, weight_decay, nesterov, ns_steps):
        buf = state["momentum_buffer"]
        buf.lerp_(p.grad, 1 - momentum)
        update = torch.lerp(p.grad, buf, momentum) if nesterov else buf
        update = _ns_project(update, steps=ns_steps, apply_matrix_scale=True)

        if weight_decay != 0:
            p.mul_(1 - lr * weight_decay)
        p.add_(update, alpha=-lr)
