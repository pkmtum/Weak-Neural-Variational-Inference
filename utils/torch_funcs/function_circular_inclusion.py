import torch

def function_circular_inclusion(field, BF_mesh, options):
    """Function to create a circular inclusion in the mesh
    Input:
    dict{Inclusion_1: dict{center_x: float, center_y: float, radius: float, value: float},
         Inclusion_2: dict{center_x: float, center_y: float, radius: float, value: float},
        ...
        Inclusion_n: dict{center_x: float, center_y: float, radius: float, value: float}}
    """
    if "flag_DC" in options.keys():
        if not options["flag_DC"]:
            if field.size() != BF_mesh[0].size():
                field.resize_(BF_mesh[0].size())
    else:
        if field.size() != BF_mesh[0].size():
            field.resize_(BF_mesh[0].size())
    for key, val in options.items():
        if key == "flag_DC":
            continue
        a = torch.sqrt((BF_mesh[0] - val["center_x"])**2 + (BF_mesh[1] -  val["center_y"])**2)
        mask = a <= val["radius"]
        field[mask] += val["value"]
    return field.flatten()

