import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        super().__init__(params, {"lr": lr})

    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)

                with torch.no_grad():
                    p.add_(
                        p.grad,
                        alpha=-lr / math.sqrt(t + 1)
                    )

                state["t"] = t + 1

        return loss


def main():
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=100)
    for t in range(10):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(f"{loss.cpu().item():.3f}")
        loss.backward()
        opt.step()

if __name__=="__main__":
    torch.manual_seed(42)
    main()


# lr = 1: start = 24.169, end = 20.0
# lr = 10: start = 24.169, end = 3.248
# lr = 100: start = 24.169, end = 0