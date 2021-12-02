HYPERPARAMETERS

Overfit Model:
-	Conv1: 16 kernels (3x3)
-	Max pool 2x2
-	Conv2: 32 kernels (3x3)
-	Dense1: 64 nodes
-	Dense2 (output): 4 nodes

Underfit Model:
-	Conv1: 4 kernels (3x3)
-	Max pool 3x3
-	Conv2: 8 kernels (3x3)
-	Dense1: 4 nodes
-	Dense2 (output): 4 nodes


ARCHITECTURE
Conv2d, relu -> MaxPool -> Conv2d, relu -> Flatten -> Dense -> Dense