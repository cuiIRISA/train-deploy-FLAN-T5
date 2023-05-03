from typing import Dict, List, Any
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def model_fn(model_dir):
    # load model and processor from model_dir
    model =  AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()
    return model.to(device), tokenizer


def predict_fn(data, model_and_tokenizer):
    # unpack model and tokenizer
    model, tokenizer = model_and_tokenizer

    # process input
    inputs = data.pop("inputs", data)
    parameters = data.pop("parameters", None)

    # preprocess
    input_ids = tokenizer(inputs, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)


    with torch.no_grad():
        # pass inputs with all kwargs in data
        if parameters is not None:
            outputs = model.generate(input_ids, **parameters)
        else:
            outputs = model.generate(input_ids)

    # postprocess the prediction
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return [{"generated_text": prediction}]