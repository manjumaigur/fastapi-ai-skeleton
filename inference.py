import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
from pydantic import BaseModel

class InputData(BaseModel):
    question: str
    context: str

def load_model(model_id: str):
    # TODO: Cache the model
    return {
        "tokenizer": DistilBertTokenizer.from_pretrained(f"model/{model_id}"),
        "qamodel": DistilBertForQuestionAnswering.from_pretrained(f"model/{model_id}"),
    }


def run(model_id: str, input_data: InputData) -> str:
    model = load_model(model_id=model_id)

    if "context" not in input_data or "question" not in input_data:
        # TODO: raise error
        pass

    inputs = model["tokenizer"](
        input_data.question, input_data.context, return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model["qamodel"](**inputs)

    answer_start_idx = outputs.start_logits.argmax()
    answer_end_idx = outputs.end_logits.argmax()

    predicted_answer_tokens = inputs.input_ids[
        0, answer_start_idx: answer_end_idx + 1
    ]

    output = model["tokenizer"].decode(predicted_answer_tokens)

    return output
