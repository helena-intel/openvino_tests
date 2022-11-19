from huggingface_hub import HfApi
from optimum.intel.openvino import *
from transformers import AutoTokenizer, pipeline

MODEL_NAMES = {
    "hf-internal-testing/tiny-random-bert": "OVModelForMaskedLM",
    "hf-internal-testing/tiny-random-distilbert": "OVModelForSequenceClassification",
    "hf-internal-testing/tiny-random-mbart": "OVModelForSeq2SeqLM",
    "hf-internal-testing/tiny-random-roberta": "OVModelForQuestionAnswering",
    "hf-internal-testing/tiny-random-gpt2": "OVModelForCausalLM",
    "hf-internal-testing/tiny-random-t5": "OVModelForSeq2SeqLM",
    "hf-internal-testing/tiny-random-bart": "OVModelForSeq2SeqLM",
}

TASKS = {"OVModelForMaskedLM": "fill-mask", "OVModelForSequenceClassification": "text-classification", "OVModelForQuestionAnswering":"question-answering", "OVModelForCausalLM": "text-generation"}

for model_id, model_class_str in MODEL_NAMES.items():
    info = HfApi().model_info(model_id)
    task = info.pipeline_tag
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model_class_str = info.transformersInfo["auto_model"].replace("AUTO", "OV")
    print(model_id, task, model_class_str)
    print("...................................................")
    model_class = eval(model_class_str)
    model = model_class.from_pretrained(model_id, from_transformers=True)

    input_text = "hello world"
    if model_class_str == "OVModelForQuestionAnswering":
        input_text = [input_text] * 2
    elif model_class_str == "OVModelForMaskedLM":
        input_text = [f"{input_text} {tokenizer.mask_token}"]
    else:
        input_text = [input_text]

    if model_class_str in TASKS:
        task = TASKS[model_class_str]
        pipe = pipeline(task, model=model, tokenizer=tokenizer)
        pipe(*input_text)
