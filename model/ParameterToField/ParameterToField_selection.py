def ParameterToField_selection(field_kind, field_options):
    all_fields = ["Chebyshev", "Bump", "Constant", "Linear", "PiecewiseConstant", "RBF", "Polynomial", "FullField", "ConstantTriangle"]
    field_classes = {
        "Chebyshev": "model.ParameterToField.class_ChebyshevPolynomials.ChebyshevPolynomials",
        "Bump": "model.ParameterToField.class_BumpBasisFunction.BumpBasisFunction",
        "Constant": "model.ParameterToField.class_ConstantField.ConstantField",
        "Linear": "model.ParameterToField.class_LinearBasisFunctionTriangle.LinearBasisFunction",
        "PiecewiseConstant": "model.ParameterToField.class_PiecewiseConstantBasisFunction.PiecewiseConstantBasisFunction",
        "RBF": "model.ParameterToField.class_RadialBasisFunction.RadialBasisFunction",
        "Polynomial": "model.ParameterToField.class_PolynomialBasisFunction.PolynomialBasisFunction",
        "FullField": "model.ParameterToField.class_FullField.FullField",
        "ConstantTriangle": "model.ParameterToField.class_ConstantBasisFunctionTriangle.ConstantBasisFunctionTriangle"
    }

    if field_kind in field_classes:
        field_class = field_classes[field_kind]
        module_name, class_name = field_class.rsplit('.', 1)
        module = __import__(module_name, fromlist=[class_name])
        field_class = getattr(module, class_name)
        return field_class(field_options)

    raise RuntimeError(("Please select an implemented prior for x. You selected \"{}\". \n"
                        "Currently available are: {}. \n").format(field_kind,
                                                                  ', '.join(map(str, all_fields))))
