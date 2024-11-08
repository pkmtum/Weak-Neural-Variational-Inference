import torch

def slice_it(parameters, indices):
    """Slices a list at specific indices into constituent lists.
    """
    # for trivial case (no parameters given (for e.g. constant fields))
    if parameters is None:
        return [torch.tensor([]) for i in range(len(indices) + 1)]
    # IDEA: I always add beginning and end, so I can get empty lists if the end index is already included
    # Watch out: I usually hand in self.indices, so indices.append or indices.insert changes not only a local copy,
    # but also the self.indices you hand in.
    indices = [0] + indices + [parameters.size(dim=-1)]

    # indices.append(len(parameters))
    # indices.insert(0, 0)

    # # check that it has an end ...
    # if indices[-1] != len(parameters):
    #     indices.append(len(parameters))
    # # ... and a beginning
    # if indices[0] != 0:
    #     indices.insert(0, 0)

    # sanity checks
    assert len(indices) <= parameters.size(dim=-1), "Please make sure your indices can split the list."
    assert max(indices) - 1 <= parameters.size(dim=-1), "Please make sure your indices can split the list."

    # slice.
    return [parameters[..., indices[i]:indices[i + 1]] for i in range(len(indices) - 1)]


