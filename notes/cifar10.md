# CIFAR-10 Project Notes

### Run 001 — SmallCNN baseline

**Question:** Can a two-block CNN learn useful features for CIFAR-10 classification?

**Setup**
- Data: 45,000 training / 5,000 validation images; split seed 0
- Model: two convolution blocks followed by a fully connected classifier
- Batch size: 64
- Loss: cross-entropy
- Optimizer and learning rate: [fill in]
- Epochs: [fill in]
- Augmentation: none; fixed normalization only

**Results**
- Best validation accuracy: [fill in]
- Training accuracy at the selected epoch: [fill in]
- Training time: [fill in]
- Parameter count: [fill in]

**Interpretation:** 

**Next experiment:** 

---

## Architecture notes

### SmallCNN baseline

- **Padding:** Adds pixels around an image before convolution. Here, with a `3×3` kernel, stride 1, and padding 1, a `32×32` feature map remains `32×32`. Without padding, it becomes `30×30`.
- **ReLU:** Adds nonlinearity between layers. Without nonlinear activations, just be a linear mapping.
- **Max pooling:** Reduces spatial resolution and the computation needed by later layers. It also discards spatial detail, so we have to be deliberate about when to use it.

### Choosing an architecture

1. Start with a credible baseline and verify the data and training pipeline.
2. Compare training and validation performance. If both are poor, investigate capacity, optimization, and data quality. If training is strong but validation is weak, investigate generalization before increasing capacity.
3. Change one meaningful factor at a time 
4. Measure computational cost alongside accuracy, for example parameter count, training time, memory use, and inference latency.
5. When application performance is the goal, just compare against established architectures 