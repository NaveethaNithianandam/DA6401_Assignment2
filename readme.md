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

### Conclusion

While the base VGG11 provides strong feature extraction, the customizations, particularly **high dropout and staged training** are critical for improving generalization. The observed mild overfitting and training instability directly motivate these design choices, making them both theoretically sound and empirically justified.


## 1.2. Architectural Reasoning (Localization Task – Encoder Adaptation)

For the localization task, the convolutional backbone from the trained **VGG11** classification model is reused as a feature extractor. This leverages the rich hierarchical features already learned for pet breed recognition. The encoder weights are **frozen during localization training**, and only the localization head is trained.

- The convolutional backbone already captures **general visual features** such as edges, textures, and object parts, which are equally useful for localization.
- Bounding box prediction is a **lower-dimensional regression task** (4 coordinates), which does not require re-learning deep visual representations.
- Freezing prevents **destructive updates** to previously learned features, especially given the smaller effective supervision signal compared to classification.
- Training remains **stable from the beginning**, with smooth loss reduction and no sudden spikes.
- The model achieves **consistent improvements in IoU**, indicating that the pretrained features are already well-suited for spatial understanding.
- No significant gap between training and validation IoU suggests **good generalization**.
- When the encoder is unfrozen (or when higher learning rates are used), training becomes unstable and performance degrades, indicating that fine-tuning can **disrupt useful pretrained representations**.

### Conclusion

Freezing the encoder provides a strong initialization and ensures stable training while avoiding overfitting and instability. The observed results confirm that pretrained classification features transfer effectively to localization, making fine-tuning unnecessary and even harmful in this setting.

## 1.3. Loss Formulation (Segmentation)

For the segmentation task, we formulate the problem as a **per-pixel classification task**, where each pixel is assigned a class label. Based on this, we use **Cross-Entropy Loss** as the primary training objective.

Cross-Entropy Loss is well-suited for segmentation because it directly optimizes the probability distribution over classes at each pixel, providing **stable gradients and efficient convergence**. It ensures that the model learns strong local (pixel-wise) predictions, which are essential for accurate segmentation.

Although Cross-Entropy does not explicitly optimize spatial overlap metrics such as Intersection-over-Union (IoU), we monitor **mean IoU (mIoU)** during training to ensure that improvements in pixel-wise accuracy translate into better structural segmentation.
The training results validate this choice:

- Validation loss decreases rapidly from **0.2070 to ~0.13 within the first 5 epochs**, showing fast and stable convergence.
- Validation mIoU improves consistently from **0.7430 to a peak of 0.8202**, indicating strong spatial learning.
- The gap between training and validation performance remains small (final train mIoU ≈ 0.863 vs val mIoU ≈ 0.820), suggesting **good generalization**.
- Even in later epochs, where loss improvements plateau, mIoU continues to improve slightly, showing that the model refines spatial predictions over time.

### Conclusion

Cross-Entropy Loss is justified as it:
- Provides **stable and efficient optimization**
- Aligns naturally with the **pixel-wise classification formulation**
- Achieves strong empirical performance (**0.8202 validation mIoU**)

Overall, the results demonstrate that Cross-Entropy, when combined with a strong architecture, is sufficient to produce **accurate and spatially consistent segmentation outputs**.

## 1.4. Multi-Task Output
<img width="1630" height="587" alt="OP_asg2" src="https://github.com/user-attachments/assets/b00efcc1-4d9d-4508-a535-eb96166295e8" />

W&B Report : https://wandb.ai/da25d005-indian-institute-of-technology-madras/da6401_a2/reports/Untitled-Report--VmlldzoxNjQ0NTMyMA?accessToken=kclh42cxpk1ehyapt6o318jex7tb9b8zkuk04arr2fqeqrn5ufq3nf54aumka7l9
