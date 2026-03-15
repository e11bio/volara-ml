.. _sec_overview:

Overview
========

What is `volara-ml`?
^^^^^^^^^^^^^^^

`volara-ml` is an ML library for blockwise prediction of large `N` dimensional datasets.
It uses the framework of the `volara` `BlockwiseTask` class to provide a simple interface for running
models in a variety of blockwise settings from serial execution without multiprocessing for debugging
to distributed execution on compute clusters via `volara.workers`.
