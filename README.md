# Research Paper Analysis & Classification Pipeline

This repository contains a sample pipeline for classifying research abstracts.
It compares several biomedical language models using
[Hugging Face Transformers](https://github.com/huggingface/transformers) and
reports the best performing one. The candidate models include PubMedBERT,
BioBERT, SciBERT, ClinicalBERT and BioGPT.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place the training data in a directory structure like:
   ```
   Dataset/
       Cancer/
           <pmid>.txt
       Non-Cancer/
           <pmid>.txt
   ```
   Each text file contains three lines with the PMID, title, and abstract:
   ```
   <ID:12345678>
   Title: Example title
   Abstract: The abstract text...
   ```

3. Run the pipeline:
   ```bash
   python pipeline.py
   ```

The script trains each candidate model, reports their evaluation metrics and
extracts diseases mentioned in each abstract using `spaCy`/`scispaCy`. The model
with the highest F1-score is selected and used for inference.

To analyse the dataset without running the full training pipeline you can
extract all disease mentions directly:

```bash
python disease_extraction.py
```
This generates a `disease_mentions.csv` file listing the PMID, label and the
diseases detected in every abstract.

## Example Output

Running `python pipeline.py` prints evaluation metrics for each candidate model
and reports which one performed best. An abbreviated example is shown below:

```
PubMedBERT Performance:
Accuracy: 90%
F1-score: 0.84

BioBERT Performance:
Accuracy: 92%
F1-score: 0.86

Best Model: BioBERT
Confusion Matrix:
                Predicted Cancer  Predicted Non-Cancer
Actual Cancer                 350                    50
Actual Non-Cancer              30                   570

The script also prints multi-label predictions with confidence scores and the
diseases detected for each abstract in the evaluation set:

```
{
  "abstract_id": "12345",
  "predicted_labels": ["Cancer"],
  "confidence_scores": {
    "Cancer": 0.92,
    "Non-Cancer": 0.85
  },
  "extracted_diseases": ["Lung Cancer", "Breast Cancer"]
}
```

Performance Improvement Analysis:
* Accuracy increased by 7% after fine-tuning.
* Reduction in false negatives, improving model reliability.
* Fine-tuned model provides better classification confidence.
```

## Streaming and Batch Processing

`pipeline.py` supports batch inference and Kafka streaming. Predictions for a
DataFrame are now processed in batches, reducing overhead when analysing many
papers at once. Enable streaming by setting the `stream` flag in the `Config`
dataclass. Incoming papers are consumed from `kafka_input_topic`, processed in
batches, and the results are published to `kafka_output_topic`.

## Structured Output & Citation Analysis

Predictions are saved to `outputs/structured_outputs.json` for easy post-
processing. Each entry includes the model confidences, extracted diseases and a
citation analysis generated with a lightweight LLM (default `google/flan-t5-base`).
The LLM summarises the context around any `[CITATION]` markers to explain the
purpose of referenced works.

## LangChain Query Engine

After generating `outputs/structured_outputs.json` you can explore the results interactively using LangChain:

```bash
python langchain_query.py
```

Ask questions about the predictions or detected diseases in natural language. Type `exit` to quit the session.


## Autogen Multi-Agent Flow

`multiagent_flow.py` demonstrates how to orchestrate the pipeline steps with the Autogen framework. Agents collaborate in a group chat to load data, train models and evaluate results.

Run the flow with:

```bash
python multiagent_flow.py
```

Ensure an OpenAI API key is configured in your environment so the agents can communicate.


## Notes

- This example assumes the `en_core_sci_sm` spaCy model is installed.
- Adjust the `Config` dataclass in `pipeline.py` to change hyperparameters or paths.
- Actual training requires a GPU and may take considerable time depending on the
  dataset size.
- The pipeline can be toggled between single- and multi-label classification by
  setting `problem_type` in the `Config` dataclass.

