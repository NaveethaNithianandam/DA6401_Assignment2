## 1.1. Architectural Reasoning (Classification Task)

The model is based on a custom implementation of **VGG11**. This enables hierarchical feature extraction, which is well suited for fine-grained tasks like pet breed classification where subtle visual differences must be captured.

### Key Customizations and Justification

- **High Dropout in Classifier (p = 0.6):**  
  Introduced to combat overfitting due to the large fully connected layers in VGG11. Empirically, the model achieved very high training accuracy (~97%) but significantly lower validation accuracy (~61%), indicating mild overfitting. This justifies the need for strong dropout to improve generalization.

- **Two-Phase Training (Freeze → Fine-tune):**  
  Initially freezing the backbone stabilizes learning and allows the classifier to adapt to extracted features. Upon unfreezing, validation accuracy improves significantly, showing that fine-tuning enhances feature representations. However, a temporary drop in performance at the transition indicates sensitivity to large updates, motivating careful learning rate control.

- **Reduced Learning Rate during Fine-tuning:**  
  A lower learning rate is used after unfreezing to prevent disrupting learned features. Empirically, high learning rates caused instability and accuracy drops, reinforcing the importance of gradual fine-tuning.

- **Weight Decay and Gradient Clipping:**  
  These are used to regularize training and ensure stability. The observed gap between training and validation performance further justifies their inclusion to control model complexity.

### Summary

While the base VGG11 provides strong feature extraction, the customizations, particularly **high dropout and staged training** are critical for improving generalization. The observed mild overfitting and training instability directly motivate these design choices, making them both theoretically sound and empirically justified.
