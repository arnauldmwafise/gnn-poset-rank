"""
posetrank: Diagnostic and predictive tools for poset-aware graph neural ranking.

Companion software for "When Do Graph Neural Rankers Need Partial Orders?
A Diagnostic and Predictive Framework for Poset-Aware Ranking".

Subpackages
-----------
diagnostics : Cycle Inconsistency Rate, Transitive Redundancy Rate, and
    Structural Incomparability Rate -- a pre-modeling audit of how
    totally-orderable a comparison dataset is.
data : Loaders for every dataset used in the paper (citation networks,
    Wikipedia hyperlinks, co-purchase networks, LETOR learning-to-rank
    data, skyline benchmarks, and the synthetic Pareto-dominance
    generator), each returning genuinely directed, verified data.
models : The shared directed dual-stream encoder and the two prediction
    heads compared throughout the paper (TotalOrderScoreModel,
    PartialOrderModel).
training : Shared pair-sampling and train/eval routines used by every
    experiment.
experiments : Scripts reproducing each table in the paper.
"""

__version__ = "1.0.0"
