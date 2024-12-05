import os

def get_legacy():
    return os.environ.get("TF_USE_LEGACY_KERAS", None)


def get_format(args):
    return "" if args.format is None else args.format


def validate_args(args):
    is_legacy, format = get_legacy(), get_format(args)
    if format == "keras" and is_legacy == "1":
        raise RuntimeError("Invalid format when using `TF_USE_LEGACY_KERAS`.")
    
    if is_legacy is None and format != "keras":
        raise RuntimeError("Invalid format. Expected `keras` format.")

def save_model_tf(
    model, args, model_name, data, framework, version
):
    is_legacy, format = get_legacy(), get_format(args)
    name = '-'.join((data, model_name, framework + version)) + '.' + format
    kwargs = {"save_format": format} if is_legacy == "1" and format == "h5" else {}
    model.save(name, **kwargs)
