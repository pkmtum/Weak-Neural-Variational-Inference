from utils.function_load_from_list import load_from_list

def PDE_selection(PDE_kind, PDE_options):
    prior_classes = {
        "Hook": "model.PDEs.class_Hook.Hook",
        "NeoHook": "model.PDEs.class_NeoHook.NeoHook"
    }
    return load_from_list(prior_classes, PDE_kind, PDE_options)