"""
SNN Architecture — snn-pareto
Week 2: LIF neurons replacing ReLU, surrogate gradient training.
Placeholder — full implementation built in Week 2 notebook.
"""

# TO BE IMPLEMENTED IN WEEK 2
# Key components:
#   snntorch.Leaky (beta=0.9, threshold=1.0)
#   snntorch.surrogate.fast_sigmoid (slope=25)
#   Poisson spike encoding via snntorch.spikegen.rate
#   Forward pass returns spike tensor of shape [T, B, 10]
