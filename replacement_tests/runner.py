"""
This file run tests and benchmarks of the replaced operations.
"""
from test_classes import BackwardNotSupported, ReplacementIsEquivalent, GradientIsEquivalent, GradientIsNull
from ops import *
from inputs import *

if __name__ == "__main__":
    tests = []
    # test replace index_select by gather
    tests += [
        BackwardNotSupported("index_select", IndexSelectModel(), index_select_input, 0),
        ReplacementIsEquivalent("index_select/gather",
                                IndexSelectModel(), index_select_input,
                                GatherModel(), index_select_input),
        GradientIsEquivalent("index_select/gather",
                                IndexSelectModel(), index_select_input, 0,
                                GatherModel(), index_select_input, 0)
    ]

    # test replace 

    for t in tests:
        t.run()


