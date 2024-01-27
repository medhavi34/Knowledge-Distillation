In the quest for efficient neural networks, our research harnesses distillation methods to balance computational efficiency with superior performance. By transitioning from large, complex networks to more streamlined models, we aim to reduce the high computational overhead.
1. Knowledge Distillation (KD) in Simple Neural Networks:
The objective function used in this step: 
𝐿𝑜𝑠𝑠=𝛼∗𝐻(𝑝,𝑞′)∗𝑡^2+(1−𝛼)  ∗𝐻(𝑝,𝑞)
where 𝑝,𝑞,𝑞’ are the student probabilities, actual targets and the soft probabilities from the teacher respectively, 𝐻(𝑝,𝑞) is the cross-entropy loss between the student logits and the actual targets, and 𝛼 is the weight parameter 
We employ a large network with dual 1200-unit hidden layers , 50% dropout in hidden layers, and 20% dropout in the input layer, training it for 50 epochs on MNIST for robust regularization.A smaller model with two 30-unit layers, sans dropout, is distilled from the larger for 25 epochs.
Distillation is optimized by tuning alpha, 𝛼, and temperature, 𝑇, hyperparameters. Transfer to the smaller model.

2. Knowledge Distillation for Unknown Targets:
Train the student network on a reduced MNIST transfer dataset using soft probabilities from the teacher model, optimized by alpha and temperature settings.
3. Self and Reverse KD in ResNet18 and ResNet50 networks:
The objective function used in this step:
𝐿𝑜𝑠𝑠= 𝛼∗𝐷_𝐾𝐿 (𝑝′,𝑞′)∗𝑡^2+(1−𝛼)∗𝐻 (𝑝,𝑞)
Where 𝑝,𝑞,𝑝’,𝑞’ are the student probabilities, actual targets, student soft probabilities and teacher soft probabilities respectively. 𝐷_𝐾𝐿 is the KL divergence loss. For all models, consider 𝛼=0.5 and 𝑇 = 11. 
ResNet18 and ResNet50 are trained on CIFAR10, serving as teacher models.
ResNet50 is distilled from ResNet18 using its soft probabilities.
A second ResNet50 undergoes self-distillation with guidance from the first ResNet50. ResNet18 is self-distilled using its own soft outputs to train a subsequent ResNet18 model.

4. Knowledge Distillation in ConvNext Model and Transformer model
The ConvNext Model (tiny) is trained on the APTOS blindness detection dataset, with an initial learning rate of 0.001 using the Adam optimizer, adapting the final layer to five output features for 50 epochs and halving the learning rate post-25 epochs.
Knowledge Distillation is conducted on a Swin Transformer model, employing the well-trained ConvNext as a teacher, with distillation parameters set at 𝛼=0.25 and 𝑇=11.
The ConvNext model also undergoes self-distillation using identical alpha and temperature settings, enabling it to refine and consolidate its learned representations further.


