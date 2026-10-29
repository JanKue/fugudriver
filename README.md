**Fugudriver** is a simple RLlib frontend for PufferDrive.

It provides RLlib-compatible environments (single- and multi-agent) which wrap around the PufferDrive API, a RLlib (RLModule) implementation of a default model for said environments, and sample scripts for training such models with RLlib/Ray.

### Instructions

1. A Python 3.9+ environment is required. Install the base requirements: `pip install torch "ray[rllib,tune]"`.
2. Install PufferDrive following the instructions at https://emerge-lab.github.io/PufferDrive/getting-started.html
    * *except* use the patched (submodule) version of PufferDrive instead of the original repo
    * this step includes downloading the data and preprocessing it into binaries
3. Run the code as Python modules, e.g. `python -m fugudriver.test_single_agent`

### Notes on PufferDrive submodule

A patched version of PufferDrive is provided as a submodule. This should be used instead of the official PufferDrive 2.0 version.

This version extends the Python API and fixes some small bugs in the original.
