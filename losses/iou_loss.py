"""Custom IoU loss
"""

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """IoU loss for bounding box regression.

    Boxes are in (x_center, y_center, width, height) format, pixel space.

    L_IoU = 1 - IoU, which is in [0, 1]:
        0 -> perfect overlap, 1 -> zero overlap.

    Supports reduction types: "none", "mean" (default), "sum".

    Implementation note on numerical stability
    ------------------------------------------
    Predicted w/h can temporarily be negative early in training (before
    Softplus saturates). We clamp areas to >= 0 so that the IoU computation
    does not produce NaN gradients in those edge cases.
    """

    _VALID_REDUCTIONS = {"none", "mean", "sum"}

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        if reduction not in self._VALID_REDUCTIONS:
            raise ValueError(
                f"reduction must be one of {self._VALID_REDUCTIONS}, got '{reduction}'"
            )
        self.eps = eps
        self.reduction = reduction

    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """[cx, cy, w, h] -> [x1, y1, x2, y2]."""
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        return torch.stack([cx - w / 2, cy - h / 2,
                            cx + w / 2, cy + h / 2], dim=1)

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute IoU loss.

        Args:
            pred_boxes:   [B, 4]  (cx, cy, w, h) in pixel space.
            target_boxes: [B, 4]  same format.

        Returns:
            loss: scalar if reduction in {"mean","sum"}, [B] if "none".
                  Always in [0, 1].
        """
        pred_xyxy   = self._cxcywh_to_xyxy(pred_boxes)
        target_xyxy = self._cxcywh_to_xyxy(target_boxes)

        inter_x1 = torch.max(pred_xyxy[:, 0], target_xyxy[:, 0])
        inter_y1 = torch.max(pred_xyxy[:, 1], target_xyxy[:, 1])
        inter_x2 = torch.min(pred_xyxy[:, 2], target_xyxy[:, 2])
        inter_y2 = torch.min(pred_xyxy[:, 3], target_xyxy[:, 3])

        inter_area = (inter_x2 - inter_x1).clamp(min=0) * \
                     (inter_y2 - inter_y1).clamp(min=0)   

   
        area_pred   = (pred_boxes[:, 2]   * pred_boxes[:, 3]).clamp(min=0)
        area_target = (target_boxes[:, 2] * target_boxes[:, 3]).clamp(min=0)

        union = area_pred + area_target - inter_area + self.eps

        iou  = inter_area / union           
        loss = 1.0 - iou                    

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else: 
            return loss