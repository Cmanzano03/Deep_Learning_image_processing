---
---
**Phase 0: Preprocessing & Normalization**

As the images were being fed raw directly to the model in the template, we added a standard normalization step (dividing all pixel values by 255.0). **Why we did this:** Raw pixel values (0-255) cause massive, unstable weight updates during backpropagation. Normalizing them to a [0, 1] range ensures gradient stability and faster convergence.

**Phase A: Architecture Search**

The main idea in this phase was to try different base architectures to pick the best-performing one to use in the subsequent tuning phase. **Why we did this:** We needed to find the optimal balance of network capacity (depth vs. width) before tweaking advanced parameters, ensuring the "engine" was the right size for the problem.

1. **Baseline**: We used the default model given in the template (no hidden layers, but including the normalization from Phase 0) to establish a baseline for evaluating the other architectures.
    
2. **1 layer of 1048 neurons**: We tried a shallow network with a generous amount of neurons. **Why:** To see if a single massive layer could map spatial features without losing gradient strength. It did not perform well at all, as it overfitted very early.
    
3. **3 layers (512 -> 256 -> 128)**: Bad performance; it heavily overfitted to the majority class, constantly predicting "Building". **Why:** Too much depth without regularization caused the network to lose spatial context and default to the safest statistical guess.
    
4. **4 layers (256 -> 64 -> 32 -> 16)**: This was the best-performing architecture. While not perfect, the depth of the network allowed it to start identifying shapes, though it still struggled to differentiate between two objects with similar geometries.
    

**Phase B: Greedy Optimization (The True Winner)**

In this phase, we tuned the winning architecture from Phase A using Greedy Optimization (Coordinate Descent). We kept only the best-performing option from each step and discarded the rest. We increased the epochs to 50 and added an Early Stopping callback to prevent overfitting.

1. **Swish Activation**: Performance did not improve significantly, but we kept it. **Why we tried it:** Standard ReLU activations block negative values entirely, often leading to "dead neurons". Swish allows small negative gradients to flow, which helps the network learn more complex, subtle image patterns.
    
2. **he_normal initialization**: Initially tested as step 1, but most neurons died due to ReLU outputting 0. Paired with Swish, we got unexpectedly good results (val_accuracy = 0.528). **Why we tried it:** Standard initialization can cause exploding or vanishing gradients. `he_normal` is mathematically optimized for non-linearities like ReLU/Swish, keeping the variance of the weights stable from epoch 1.
    
3. **Focal loss function**: The model improved quickly but hit a ceiling around ~0.52 accuracy. **Why we tried it:** Our dataset suffers from extreme class imbalance (e.g., thousands of "Buildings" vs. only 111 "Helipads"). Standard Categorical Crossentropy gets overwhelmed by the majority class. Focal loss introduces a mathematical $\gamma$ factor that down-weights easy, common examples and forces the network to penalize errors on hard-to-classify, minority classes.
    
4. **SGD optimizer with momentum**: This combination broke the ceiling, reaching 0.5584 validation accuracy at epoch 29. **Why we tried it:** While standard SGD can get stuck in local minima, adding _Momentum_ accelerates gradients in the right direction and dampens oscillations, providing a smoother and deeper convergence. _(Note: Due to a bug discovered later in Phase C, this model ultimately proved to be our best-performing, most robust solution on unseen data)._
    
5. **Z-score normalization**: Results were unremarkable, so it was discarded. **Why we tried it:** We wanted to fully exploit the power of the Swish activation function. By centering the pixel distribution around zero (mean 0, unit variance), we hypothesized Swish would process the negative inputs more effectively than standard /255.0 scaling.
    

**Phase C: The Resolution Filter & The Normalization Bug (Runner-Up)**

Initially, we hit an accuracy ceiling at ~0.52 and hypothesized that the network was too deep. **Why we downsized to 64x64:** We realized that overly deep, unregularized FFNNs were simply memorizing high-resolution pixel noise instead of generalizing actual shapes. Downsizing to 64x64 acts as a natural noise filter, forcing the network to focus strictly on macro-patterns and core geometries. We used a minimalist architecture: `64x64x3 -> Dense(512), swish, he_normal -> Dense(13), softmax`.

_The Plot Twist:_ Initially, this Phase C model appeared to be our absolute best. However, we discovered a critical bug in our pipeline: **we had forgotten to apply the Phase 0 pixel normalization (/255.0) to our validation and test sets in previous evaluations.** Once the normalization bug was corrected across all sets, Phase C still yielded highly respectable results (recovering complex minority classes like Fishing Vessels at 53.4% recall), but it failed entirely to recognize Helipads (0% recall). Upon fair re-evaluation, the deeper Phase B.4 model actually outperformed Phase C.

**Phase D: Pushing the Limits & Validation Overfitting**

Our objective here was to break the 56% accuracy barrier and recover the "dead" Helipad class without using explicit regularization (Dropout) or Data Augmentation.

1. **Upscaling & Wide Architecture**: We increased resolution to 128x128. **Why:** 64x64 was too compressed to capture the fine details of tiny objects like Shipping Containers. To process this massive increase in spatial features without adding depth (which dilutes gradients), we implemented a "Wide & Shallow" approach: a single massive Dense layer of 1024 neurons.
    
2. **Focal Loss**: Maintained from Phase B. **Why:** To prevent the massive new layer from lazily predicting only the majority classes.
    
3. **Double-Training & Manual LR Decay**: We trained the model once, loaded those pre-trained weights, and then executed a second training run with a manually halved learning rate (0.005). **Why:** With nearly 50 million parameters, the wide architecture was highly prone to severe instability and "bouncing". The LR decay acted as a stabilizer, allowing the massive model to take smaller steps and gently settle into a local minimum.
    

_The Codabench Reality Check:_ This weird double-training approach yielded our absolute highest _validation_ results by far: 59.09% accuracy, recovering Helipads (33.3% precision) and Shipping Containers (65.18% recall).

However, evaluating this model on the blind Codabench Test Set resulted in a terrible drop to **50.06% Accuracy**.

This ~9-point drop is a textbook example of **Validation Overfitting**. By extensively tuning hyper-parameters and executing consecutive training runs on an unregularized 50M-parameter model, the network completely memorized the validation set. It built a highly fragile representation that failed terribly on unseen test data, empirically proving the absolute limits of pure architectural tuning and perfectly setting the stage for Phase 2 (Regularization).

# Results obtained
After a comprehensive evaluation and correcting the Phase 0 normalization bug across the validation and test splits, the Phase B.4 model emerged as the definitive winner of this unregularized phase.

While the other architectures provided valuable insights, they ultimately failed on unseen data or specific minority classes: the shallow 64x64 model (Phase C) completely lost the ability to detect 'Helipads', and the massive 1024-neuron model (Phase D) suffered from severe validation overfitting, memorizing the training data rather than learning generalizable shapes.

Therefore, the tuned deep architecture from Phase B.4 (4 hidden layers, Swish activation, he_normal initialization, Focal Loss, and SGD with momentum) proved to be the most robust solution. It struck the optimal mathematical balance between network capacity and true generalization on unseen data. By successfully managing the extreme class imbalance without relying on explicit regularization techniques, Phase B.4 stands as our most reliable and effective baseline as we transition into Phase 2. Concrete results obtained:

* Mean Accuracy : 51.04
* Mean Precission: 50.71
* Mean recall: 48.36
