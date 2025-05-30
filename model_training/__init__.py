from model_training.base_pl import Vanilla


def get_training_method(method_name):
    methods = {
        "Vanilla": Vanilla,
    }
    return methods[method_name]
