def load_from_list(classes, kind, options):
    """
    This function helps to find the correct prior to your input strings
    :param classes: dict: mapping of prior kinds to their corresponding classes
    :param kind: str: e.g. "MVN" for Multivariate Normal
    :param options: dict with options like mean and sigma
    :return:
    """
    if kind in classes:
        class1 = classes[kind]
        module_name, class_name = class1.rsplit('.', 1)
        module = __import__(module_name, fromlist=[class_name])
        class1 = getattr(module, class_name)
        return class1(options)

    raise RuntimeError(("Please select an implemented prior for x. You selected \"{}\". \n"
                        "Currently available are: {}. \n").format(kind,
                                                                  ', '.join(map(str, classes.keys()))))
