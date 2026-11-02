"""Quilt — stitch spare machines into one language model.

Quilt runs a transformer whose layers are spread across a mesh of ordinary
devices. Each device holds a contiguous *patch* of layers; activations are
handed from patch to patch over the network; the conversation journal is
replicated to every member so the mesh survives the loss of the device that
is driving a session.
"""

__version__ = "0.1.0"
