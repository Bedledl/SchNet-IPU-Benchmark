"""
This file run tests and benchmarks of the replaced operations.
"""
from test_classes import BackwardNotSupported, ReplacementIsEquivalent, GradientIsEquivalent, GradientIsDifferentOnIPU
from ops import *
from inputs import *

if __name__ == "__main__":
    tests = []
    tests += [
        GradientIsEquivalent("Topk CPU vs IPU", TopKDistance(), topk_input, 0,
                                                TopKDistance(), topk_input, 0)
    ]

    # test replace index/index_select by gather
    tests += [
        BackwardNotSupported("index", IndexModel(), index_select_input, 0),
        ReplacementIsEquivalent("index/gather",
                                IndexModel(), index_select_input,
                                GatherModel(), index_select_input),
        GradientIsEquivalent("index/gather",
                             IndexModel(), index_select_input, 0,
                             GatherModel(), index_select_input, 0),

        BackwardNotSupported("index_select", IndexSelectModel(), index_select_input, 0),
        ReplacementIsEquivalent("index_select/gather",
                                IndexSelectModel(), index_select_input,
                                GatherModel(), index_select_input),
        GradientIsEquivalent("index_select/gather",
                                IndexSelectModel(), index_select_input, 0,
                                GatherModel(), index_select_input, 0)
    ]

    # replace softplus by log(1+e^x)
    tests += [
        BackwardNotSupported("torch.nn.functional.softplus", TorchSoftplusModel(), softplus_input, 0),
        ReplacementIsEquivalent("softplus/log(1+exp(x))",
                                TorchSoftplusModel(), softplus_input,
                                SoftplusLogExpModel(), softplus_input),
        GradientIsEquivalent("softplus/log(1+exp(x))",
                             TorchSoftplusModel(), softplus_input, 0,
                             SoftplusLogExpModel(), softplus_input, 0)
    ]

    # test replace norm and linalg norm by pow/sum/sqrt
    tests += [
        BackwardNotSupported("Norm", NormModel(), norm_input, 0),
        GradientIsEquivalent("LinalgNormIPUvs.CPU",
                             LinalgNormModel(), norm_input, 0,
                             LinalgNormModel(), norm_input, 0),
        ReplacementIsEquivalent("norm/linalg",
                                NormModel(), norm_input,
                                LinalgNormModel(), norm_input),
        GradientIsEquivalent("norm/linalg",
                             NormModel(), norm_input, 0,
                             LinalgNormModel(), norm_input, 0)
    ]

    # test the cosinus and the where operation in the Cutoff module
    tests = [
        ReplacementIsEquivalent("cosineCutoff",
                                CosineCutoffCPUModel(), cutoff_input,
                                CosineCutoffIPUModel(), cutoff_input),
        GradientIsEquivalent("cosineCutoff",
                             CosineCutoffCPUModel(), cutoff_input, 0,
                             CosineCutoffCPUModel(), cutoff_input, 0),

        #BackwardNotSupported("Where", WhereInPlaceModel(), cutoff_input, 0),
        ReplacementIsEquivalent("WhereInPlace/WhereNotInPlace",
                                WhereInPlaceModel(), cutoff_input,
                                WhereNotInPlaceModel(), cutoff_input),
        GradientIsEquivalent("WhereInPlace/WhereNotInPlace",
                                WhereInPlaceModel(), cutoff_input, 0,
                                WhereNotInPlaceModel(), cutoff_input, 0)
    ]


    for t in tests:
        t.run()


