def BCMask_selection(field_kind, field_options):
    all_fields = ["all_zero", "bottom_left_zero", "bottom_zero", "bottom_top_zero", "top_zero", "None"]
    field_classes = {
        "all_zero": "model.BCMasks.class_BCMask_all_zero.BCMask_all_zero",
        "bottom_left_zero": "model.BCMasks.class_BCMask_bottom_left_zero.BCMask_bottom_left_zero",
        "bottom_zero": "model.BCMasks.class_BCMask_bottom_zero.BCMask_bottom_zero",
        "top_bottom_zero": "model.BCMasks.class_BCMask_top_bottom_zero.BCMask_top_bottom_zero",
        "top_zero": "model.BCMasks.class_BCMask_top_zero.BCMask_top_zero",
        "None": "model.BCMasks.class_BCMask_None.BCMask_None",
        "all_zero_value": "model.BCMasks.class_BCMask_all_zero_value.BCMask_all_zero_value",
        "bottom_left_zero_value": "model.BCMasks.class_BCMask_bottom_left_zero_value.BCMask_bottom_left_zero_value",
        "bottom_zero_value": "model.BCMasks.class_BCMask_bottom_zero_value.BCMask_bottom_zero_value",
        "bottom_top_zero_value": "model.BCMasks.class_BCMask_bottom_top_zero_value.BCMask_bottom_top_zero_value",
        "top_zero_value": "model.BCMasks.class_BCMask_top_zero_value.BCMask_top_zero_value",
        "left_zero": "model.BCMasks.class_BCMask_left_zero.BCMask_left_zero"
    }

    if field_kind in field_classes:
        field_class = field_classes[field_kind]
        module_name, class_name = field_class.rsplit('.', 1)
        module = __import__(module_name, fromlist=[class_name])
        field_class = getattr(module, class_name)
        return field_class(field_options)

    raise RuntimeError(("Please select an BC Mask that is not implemented. You selected \"{}\". \n"
                        "Currently available are: {}. \n").format(field_kind,
                                                                  ', '.join(map(str, all_fields))))
