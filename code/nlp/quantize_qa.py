import shutil
import time
from functools import partial
from pathlib import Path

import openvino.runtime
import torch
import transformers
from datasets import load_dataset, load_metric
from nncf import NNCFConfig
from optimum.intel.openvino import OVModelForQuestionAnswering
from optimum.intel.openvino.quantization import OVQuantizer
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


def compute_metric_squad(model_id, precision):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = OVModelForQuestionAnswering.from_pretrained(
        f"models/{model_id}_{precision}"
    )
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

    dataset = load_dataset("squad", split="validation[:100]")
    metric = load_metric("squad")
    predictions = []
    references = []
    for item in dataset:
        prediction = qa_pipeline(
            {"context": item["context"], "question": item["question"]}
        )
        metric_prediction = {"id": item["id"], "prediction_text": prediction["answer"]}
        metric_reference = {
            "id": item["id"],
            "answers": {
                "answer_start": item["answers"]["answer_start"],
                "text": item["answers"]["text"],
            },
        }
        predictions.append(metric_prediction)
        references.append(metric_reference)
    return metric.compute(predictions=predictions, references=references)


def preprocess_fn_qa(examples, tokenizer):
    return tokenizer(
        examples["question"],
        examples["context"],
        padding="max_length",
        max_length=128, # tokenizer.model_max_length,
        truncation=True,
    )


def quantize(model_id, dataset_name, preprocess_function, dataset_config=None):
    print(f"Quantizing {model_id}")
    if not Path(f"models/{model_id}_FP32").exists():
        ov_model = OVModelForQuestionAnswering.from_pretrained(
            model_id, from_transformers=True
        )
        ov_model.save_pretrained(f"models/{model_id}_FP32")

    model = AutoModelForQuestionAnswering.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    quantizer = OVQuantizer.from_pretrained(model)
    calibration_dataset = quantizer.get_calibration_dataset(
        dataset_name,
        preprocess_function=partial(preprocess_function, tokenizer=tokenizer),
        dataset_config_name=dataset_config,
        num_samples=50,
        dataset_split="train",
        preprocess_batch=True,
    )
    nncf_config = NNCFConfig.from_json("nncf_bert_config_squad.json")
    start_time = time.perf_counter()
    quantizer.quantize(
        save_directory=f"models/{model_id}_INT8",
        calibration_dataset=calibration_dataset,
        batch_size=2
    )
    end_time = time.perf_counter()
    duration = end_time - start_time


def evaluate(model_id):
    for precision in ["FP32", "INT8"]:
        metrics = compute_metric_squad(model_id, precision)
        for metric_name, metric_value in metrics.items():
            print(f"{model_id},{precision},{metric_name}: {metric_value:.2f}")


# model = "deepset/tinyroberta-squad2"
# model = "deepset/minilm-uncased-squad2"
# model = "deepset/roberta-base-squad2"
# model = "deepset/xlm-roberta-large-squad2"
model = "mrm8488/bert-small-finetuned-squadv2"
model = "csarron/bert-base-uncased-squad-v1"

try:
    shutil.rmtree(Path("~/.cache/huggingface").expanduser())
except FileNotFoundError:
    pass

quantize(model, "squad", preprocess_fn_qa)
evaluate(model)
evaluate(model)
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"OpenVINO version: {openvino.runtime.__version__}")
