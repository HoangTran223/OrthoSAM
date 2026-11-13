import math

import torch
import torch.nn.functional as F

from utility.loss import smooth_crossentropy
from utility.args import Args


def zeropower_via_newtonschulz5(matrix: torch.Tensor, steps: int) -> torch.Tensor:
    assert matrix.ndim == 2

    a, b, c = (3.4445, -4.7750, 2.0315)
    x = matrix.bfloat16()

    transposed = False
    if matrix.size(0) > matrix.size(1):
        x = x.t()
        transposed = True

    x = x / (x.norm() + 1e-7)

    for _ in range(steps):
        a_mat = x @ x.t()
        b_mat = b * a_mat + c * a_mat @ a_mat
        x = a * x + b_mat @ x

    if transposed:
        x = x.t()

    return x


def _orthogonality_residual(matrix: torch.Tensor) -> float:
    if matrix.ndim != 2:
        matrix = matrix.reshape(matrix.size(0), -1)
    matrix = matrix.detach().float()
    m, n = matrix.shape
    if m >= n:
        gram = matrix.transpose(0, 1) @ matrix
        eye = torch.eye(n, device=matrix.device, dtype=matrix.dtype)
        return (gram - eye).norm().item() / math.sqrt(n)
    gram = matrix @ matrix.transpose(0, 1)
    eye = torch.eye(m, device=matrix.device, dtype=matrix.dtype)
    return (gram - eye).norm().item() / math.sqrt(m)


def _ns_project(tensor: torch.Tensor, steps: int, apply_matrix_scale: bool = False) -> torch.Tensor:
    if tensor.ndim < 2:
        return tensor

    original_shape = tensor.shape
    if tensor.ndim == 2:
        matrix = tensor
    else:
        matrix = tensor.reshape(tensor.size(0), -1)

    projected = zeropower_via_newtonschulz5(matrix, steps=steps).to(dtype=tensor.dtype)
    if apply_matrix_scale:
        projected = projected * (max(1.0, projected.size(0) / projected.size(1)) ** 0.5)
    return projected.reshape(original_shape)


_OUTPUT_HEAD_MODULES = frozenset({
    "head",
    "linear_head",
    "classifier",
    "fc",
    "linear",
    "lm_head",
    "output",
    "score",
})


def _is_embedding_or_output_name(param_name: str) -> bool:
    if not param_name:
        return False

    lowered_name = param_name.lower()
    if "embed_tokens" in lowered_name or "embedding" in lowered_name or "embeddings" in lowered_name or "lm_head" in lowered_name:
        return True

    parts = lowered_name.split(".")
    if not parts or parts[-1] not in ("weight", "bias"):
        return False

    parts = parts[:-1]
    while parts and parts[-1].isdigit():
        parts = parts[:-1]

    return bool(parts) and parts[-1] in _OUTPUT_HEAD_MODULES


def _should_use_orthosam(param: torch.Tensor, param_name: str) -> bool:
    return param.ndim >= 2 and not _is_embedding_or_output_name(param_name)


class OrthoSAM(torch.optim.Optimizer):

    def __init__(
            self,
            params,
            lr: float = 1e-1,
            momentum: float = 0.95,
            weight_decay: float = 1e-2,
            nesterov: bool = True,
            rho: float = 0.3,
            ns_steps: int = 5,
            eps: float = 1e-12,
            ):
        parameter_entries = list(params)
        if len(parameter_entries) == 0:
            raise ValueError("OrthoSAM received an empty parameter list.")

        named_entries = []
        for entry in parameter_entries:
            if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], str):
                name, param = entry
                named_entries.append((name, param))
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
        super(OrthoSAM, self).__init__(parameters, defaults)

        self._residual_ascent_sum = 0.0
        self._residual_ascent_count = 0
        self._residual_descent_sum = 0.0
        self._residual_descent_count = 0

        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    state["param_name"] = parameter_name_map.get(id(p), "")
                    state["use_orthosam"] = _should_use_orthosam(p, state["param_name"])
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p).detach()

    def _record_residual(self, projected: torch.Tensor, branch: str) -> None:
        matrix = projected.detach()
        if matrix.ndim != 2:
            matrix = matrix.reshape(matrix.size(0), -1)
        residual = _orthogonality_residual(matrix)
        if branch == "ascent":
            self._residual_ascent_sum += residual
            self._residual_ascent_count += 1
        else:
            self._residual_descent_sum += residual
            self._residual_descent_count += 1

    def pop_residual_stats(self) -> dict:
        ascent_mean = (
            self._residual_ascent_sum / self._residual_ascent_count
            if self._residual_ascent_count > 0 else float("nan")
        )
        descent_mean = (
            self._residual_descent_sum / self._residual_descent_count
            if self._residual_descent_count > 0 else float("nan")
        )
        total_count = self._residual_ascent_count + self._residual_descent_count
        overall_mean = (
            (self._residual_ascent_sum + self._residual_descent_sum) / total_count
            if total_count > 0 else float("nan")
        )
        stats = {
            "residual_mean": overall_mean,
            "residual_ascent": ascent_mean,
            "residual_descent": descent_mean,
            "residual_count": total_count,
        }
        self._residual_ascent_sum = 0.0
        self._residual_ascent_count = 0
        self._residual_descent_sum = 0.0
        self._residual_descent_count = 0
        return stats

    @torch.no_grad()
    def move_up(self):
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            rho = group["rho"]
            eps = group["eps"]
            denom = grad_norm + eps
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if self.state[p]["use_orthosam"]:
                    direction = _ns_project(p.grad, steps=ns_steps, apply_matrix_scale=False)
                    self._record_residual(direction, "ascent")
                    direction = direction / (direction.norm() + eps)
                else:
                    direction = p.grad / denom

                e_w = rho * direction
                p.add_(e_w)
                self.state[p]["e_w"] = e_w.detach().clone()

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
    def move_back(self):
        for group in self.param_groups:
            for p in group["params"]:
                if "e_w" in self.state[p]:
                    p.sub_(self.state[p]["e_w"])
                    del self.state[p]["e_w"]

    def step(self, model, inputs, targets, regression: bool = False):
        predictions = model(inputs)

        if regression:
            loss = F.mse_loss(predictions.squeeze(-1), targets.float())
        else:
            loss = smooth_crossentropy(predictions, targets).mean()
        loss.backward()

        self.move_up()
        self.zero_grad()

        model.module.setBatchNormTracking(False)
        out2 = model(inputs)
        if regression:
            loss_intermediate = F.mse_loss(out2.squeeze(-1), targets.float())
        else:
            loss_intermediate = smooth_crossentropy(out2, targets).mean()
        model.module.setBatchNormTracking(True)
        loss_intermediate.backward()

        self.move_back()
        if Args.grad_clip_norm != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Args.grad_clip_norm)
        self._orthosam_update_step()
        self.zero_grad()

        return loss, predictions

    @torch.no_grad()
    def _orthosam_update_step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if self.state[p]["use_orthosam"]:
                    update = _ns_project(p.grad, steps=ns_steps, apply_matrix_scale=False)
                    self._record_residual(update, "descent")
                else:
                    update = p.grad

                if wd != 0:
                    update = update.add(p, alpha=wd)

                if momentum != 0:
                    buf = self.state[p]["momentum_buffer"]
                    buf.mul_(momentum).add_(update)
                    update = update.add(buf, alpha=momentum) if nesterov else buf

                p.add_(update, alpha=-lr)
