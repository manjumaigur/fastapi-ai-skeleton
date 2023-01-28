from fastapi import FastAPI
import inference

app = FastAPI()


@app.post("/{model_id}/infer")
def infer(model_id: str, input_data: inference.InputData):
    output = inference.run(model_id=model_id, input_data=input_data)
    return {"answer": output}