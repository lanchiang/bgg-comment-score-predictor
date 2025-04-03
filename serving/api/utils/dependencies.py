from .models import model_container

def get_tokenizer():
    return model_container.tokenizer

def get_model():
    return model_container.model

def get_device():
    return model_container.device