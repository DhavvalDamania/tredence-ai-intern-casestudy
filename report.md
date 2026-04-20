
# Self-Pruning Neural Network — Results Report

## 1. Why L1 Penalty on Sigmoid Gates Encourages Sparsity

The total loss has two competing terms:The sparsity loss is the L1 norm (mean) of all gate values across every PrunableLinear layer.
Gates are produced by applying sigmoid to learnable gate_scores parameters.

L1 penalizes every non-zero gate value equally, regardless of its size.
This creates constant pressure on every gate to move toward zero.
A gate only stays active if its contribution to reducing classification loss
outweighs its constant cost in the sparsity term.

Unlike L2 which penalizes large values heavily but barely touches small ones,
L1 pushes small values all the way to zero - producing true sparsity rather
than just small weights.

The higher the lambda, the stronger this pressure. Weights that contribute
marginally to accuracy get pruned first. At very high lambda, even important
weights get pruned, causing accuracy to collapse.

## 2. Results Table

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
|--------|-------------------|-------------------|
| 0.1    | 47.22             | 74.75             |
| 0.3    | 43.60             | 80.81             |
| 0.7    | 30.85             | 88.02             |

## 3. Analysis of Lambda Trade-off

**Lambda = 0.1 (Low pressure)**
Moderate pruning. 74.75% of gates are eliminated while retaining 47.22% test accuracy.
The network removes clearly redundant weights while keeping the ones that matter.

**Lambda = 0.3 (Medium pressure)**
Pruning increases to 80.81%. Accuracy drops by ~3.6 percentage points.
The network is now pruning some weights that were marginally useful.

**Lambda = 0.7 (High pressure)**
Aggressive pruning at 88.02%. Accuracy falls sharply to 30.85%.
Sparsity pressure is now strong enough to kill weights the network needs.
This is the breaking point where accuracy degrades significantly.

## 4. Note on Sigmoid Gates and Binarization

Sigmoid gates never reach exactly zero during training - they asymptotically
approach it. To get true binary pruning, gates are binarized during evaluation:
any gate below 0.5 is set to 0, above 0.5 stays active.

This produces clean sparsity measurements but also explains the gap between
training accuracy (~88%) and test accuracy after binarization (~47% at lambda=0.1).
The network trains with soft gates but is evaluated with hard ones.

A harder gate mechanism such as Hard Concrete or straight-through estimators
during training itself would reduce this gap in future work.

## 5. Gate Distribution Plot

The gate distribution plot (gate_distribution.png) shows three histograms,
one per lambda value.

Each plot shows:
- A large spike near 0 - gates that were pushed below the 0.5 threshold
- A smaller cluster above 0.5 - gates the network chose to keep active
- The red dashed line marks the 0.5 binarization threshold

As lambda increases, the spike at 0 grows larger and the surviving cluster
shrinks - visually confirming that higher lambda produces more aggressive pruning.
