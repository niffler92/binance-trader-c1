import torch
import torch.nn as nn


def _process_labels(labels, label_smoothing):
    """Pre-process a binary label tensor, maybe applying smoothing.
    Parameters
    ----------
    labels : tensor-like
        Tensor of 0's and 1's.
    label_smoothing : float or None
        Float in [0, 1]. When 0, no smoothing occurs. When positive, the binary
        ground truth labels `y_true` are squeezed toward 0.5, with larger values
        of `label_smoothing` leading to label values closer to 0.5.
    Returns
    -------
    torch.Tensor
        The processed labels.
    """
    assert label_smoothing is not None
    labels = (1 - label_smoothing) * labels + label_smoothing * 0.5
    return labels


class BinaryFocalLoss(nn.Module):
    """
    Porting implementation from keras to pytorch.
    https://focal-loss.readthedocs.io/en/latest/generated/focal_loss.BinaryFocalLoss.html
    """

    def __init__(
        self,
        gamma=2.0,
        eps=1e-7,
        pos_weight=None,
        neg_weight=None,
        label_smoothing=0.1,
        reduction="mean",
    ):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, input, target):
        p = input
        q = 1 - p

        # For numerical stability (so we don't inadvertently take the log of 0)
        p = p.clamp(self.eps, 1)
        q = q.clamp(self.eps, 1)

        # Loss for the positive examples
        pos_loss = -(q ** self.gamma) * torch.log(p)
        if self.pos_weight is not None:
            pos_loss *= self.pos_weight

        # Loss for the negative examples
        neg_loss = -(p ** self.gamma) * torch.log(q)
        if self.neg_weight is not None:
            neg_loss *= self.neg_weight

        if self.label_smoothing is None:
            loss = torch.where(target.type(torch.BoolTensor), pos_loss, neg_loss)
        else:
            target = _process_labels(target, self.label_smoothing)
            loss = target * pos_loss + (1 - target) * neg_loss

        assert self.reduction in ("none", "mean", "sum")
        if self.reduction != "none":
            loss = (
                loss.mean(dim=-1).mean()
                if self.reduction == "mean"
                else torch.sum(loss)
            )

        return loss


CRITERIONS = {
    "l1": nn.L1Loss,
    "l2": nn.MSELoss,
    "bce": nn.BCELoss,
    "bfl": BinaryFocalLoss,
}
