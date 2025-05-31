"""Research Paper Analysis & Classification Pipeline"""

import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
import json

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model
import torch
import spacy
from kafka import KafkaConsumer, KafkaProducer


@dataclass
class Config:
    data_dir: Path = Path("Dataset")
    # Use a biomedical domain model
    model_name: str = "dmis-lab/biobert-base-cased-v1.1"
    # Candidate models to compare
    candidate_models: Dict[str, str] = None
    num_labels: int = 2
    output_dir: Path = Path("outputs")
    output_file: Path = Path("outputs/structured_outputs.json")
    max_length: int = 256
    batch_size: int = 8
    epochs: int = 3
    # allow multi-label classification if needed
    problem_type: str = "multi_label_classification"
    label_names: List[str] = None
    citation_model_name: str = "google/flan-t5-base"
    # streaming configuration
    stream: bool = False
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_input_topic: str = "papers"
    kafka_output_topic: str = "paper-results"
    stream_batch_size: int = 8


class CitationAnalyzer:
    """Analyse citation context using a seq2seq LLM."""

    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

    def analyze(self, text: str) -> str:
        prompt = (
            "Summarise the purpose of the citation in the following text:\n" + text
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=32)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


def load_data(path: Path) -> pd.DataFrame:
    """Load data either from a CSV file or directory of text files."""
    if path.is_dir():
        records = []
        categories = {"Cancer": 1, "Non-Cancer": 0}
        for category, label in categories.items():
            cat_dir = path / category
            if not cat_dir.exists():
                continue
            for file in cat_dir.glob("*.txt"):
                with open(file, encoding="utf-8", errors="ignore") as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                if not lines:
                    continue

                pmid = file.stem
                # handle optional ID line
                if lines[0].lower().startswith("<id:"):
                    pmid = lines[0].split(":", 1)[-1].strip("<>")
                    lines = lines[1:]

                title = ""
                if lines:
                    title = lines[0]
                    if title.lower().startswith("title:"):
                        title = title.split(":", 1)[-1].strip()
                    lines = lines[1:]

                abstract_lines = lines
                if abstract_lines and abstract_lines[0].lower().startswith("abstract:"):
                    abstract_lines[0] = abstract_lines[0].split(":", 1)[-1].strip()
                abstract = " ".join(abstract_lines)

                text = f"{title} {abstract}".strip()
                records.append({"pmid": pmid, "abstract": text, "label": label})
        df = pd.DataFrame(records)
    else:
        df = pd.read_csv(path)

    df = df.dropna(subset=["abstract"])
    df = df[df["abstract"].str.strip().astype(bool)]
    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column")
    df["abstract"] = df["abstract"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def clean_text(text: str) -> str:
    """Clean raw text by removing metadata and normalizing citations."""
    # normalize citation markers like [1] or [1,2]
    text = re.sub(r"\[(\d+(?:,\s*\d+)*)\]", "[CITATION]", text)

    # strip common metadata prefixes
    metadata_prefixes = ["<id:", "title:", "abstract:", "authors:", "author:", "journal:", "keywords:"]
    for prefix in metadata_prefixes:
        text = re.sub(rf"^\s*{prefix}\s*", "", text, flags=re.IGNORECASE)

    # drop standalone metadata lines that might remain
    text = re.sub(r"^(authors?|journal|keywords?):.*$", "", text, flags=re.IGNORECASE | re.MULTILINE)

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["abstract"] = df["abstract"].apply(clean_text)
    return df


def extract_diseases(nlp, text: str) -> List[str]:
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_.lower() in {"disease", "cancer"}]


def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(examples["abstract"], truncation=True, max_length=max_length)


def build_dataset(df: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(lambda x: tokenize_function(x, tokenizer, max_length))
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return dataset


def get_compute_metrics(cfg: Config):
    """Return a metric function configured for single or multi-label tasks."""
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if cfg.problem_type == "multi_label_classification":
            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(torch.tensor(logits))
            predictions = (probs > 0.5).int().numpy()
            report = classification_report(
                labels,
                predictions,
                target_names=cfg.label_names or ["label_%d" % i for i in range(cfg.num_labels)],
                output_dict=True,
            )
            cm = confusion_matrix(labels.argmax(axis=1), predictions.argmax(axis=1))
        else:
            predictions = np.argmax(logits, axis=-1)
            report = classification_report(
                labels,
                predictions,
                target_names=cfg.label_names or ["label_%d" % i for i in range(cfg.num_labels)],
                output_dict=True,
            )
            cm = confusion_matrix(labels, predictions)

        return {
            "accuracy": report["accuracy"],
            "f1": report["weighted avg"]["f1-score"],
            "confusion_matrix": cm,
        }

    return compute_metrics


def train_baseline(cfg: Config, train_dataset: Dataset, eval_dataset: Dataset):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=cfg.num_labels)
    model.config.problem_type = cfg.problem_type
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    args = TrainingArguments(
        output_dir=str(cfg.output_dir / "baseline"),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        evaluation_strategy="epoch",
        logging_steps=10,
        save_strategy="epoch",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=get_compute_metrics(cfg),
    )
    trainer.train()
    metrics = trainer.evaluate()
    return model, tokenizer, metrics


def train_lora(cfg: Config, train_dataset: Dataset, eval_dataset: Dataset):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=cfg.num_labels)
    base_model.config.problem_type = cfg.problem_type
    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_lin", "v_lin"], lora_dropout=0.05, bias="none")
    model = get_peft_model(base_model, lora_config)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    args = TrainingArguments(
        output_dir=str(cfg.output_dir / "lora"),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        evaluation_strategy="epoch",
        logging_steps=10,
        save_strategy="epoch",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=get_compute_metrics(cfg),
    )
    trainer.train()
    metrics = trainer.evaluate()
    return model, tokenizer, metrics


def batch_predict_and_extract(
    model,
    tokenizer,
    df: pd.DataFrame,
    cfg: Config,
    nlp,
    batch_size: int,
    citation_analyzer: CitationAnalyzer = None,
) -> List[Dict]:
    """Predict labels and extract diseases (and citations) for a batch."""
    results: List[Dict] = []
    for start in range(0, len(df), batch_size):
        batch_df = df.iloc[start : start + batch_size]
        texts = batch_df["abstract"].tolist()
        inputs = tokenizer(
            texts,
            truncation=True,
            max_length=cfg.max_length,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**inputs)
            if cfg.problem_type == "multi_label_classification":
                probs = torch.sigmoid(outputs.logits).numpy()
                pred_indices = [np.where(p > 0.5)[0] for p in probs]
            else:
                probs = torch.softmax(outputs.logits, dim=-1).numpy()
                pred_indices = [[int(np.argmax(p))] for p in probs]
        for i, (_, row) in enumerate(batch_df.iterrows()):
            labels = [cfg.label_names[idx] for idx in pred_indices[i]]
            confidences = {
                cfg.label_names[j]: float(probs[i][j]) for j in range(cfg.num_labels)
            }
            diseases = extract_diseases(nlp, row["abstract"])
            citation_summary = None
            if citation_analyzer is not None and "[CITATION]" in row["abstract"]:
                citation_summary = citation_analyzer.analyze(row["abstract"])
            results.append(
                {
                    "abstract_id": row["pmid"],
                    "predicted_labels": labels,
                    "confidence_scores": confidences,
                    "extracted_diseases": diseases,
                    "citation_analysis": citation_summary,
                }
            )
    return results


def structure_results(results: List[Dict], output_file: Path) -> Path:
    """Save structured results to JSON."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return output_file


def predict_and_extract(
    model,
    tokenizer,
    df: pd.DataFrame,
    cfg: Config,
    nlp,
    citation_analyzer: CitationAnalyzer = None,
) -> List[Dict]:
    """Generate predictions, diseases and citation analysis using batching."""
    return batch_predict_and_extract(
        model,
        tokenizer,
        df,
        cfg,
        nlp,
        batch_size=cfg.batch_size,
        citation_analyzer=citation_analyzer,
    )


def stream_papers(cfg: Config, model, tokenizer, citation_analyzer: CitationAnalyzer = None) -> None:
    """Consume papers from Kafka and publish predictions."""
    consumer = KafkaConsumer(
        cfg.kafka_input_topic,
        bootstrap_servers=cfg.kafka_bootstrap_servers,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
    )
    producer = KafkaProducer(
        bootstrap_servers=cfg.kafka_bootstrap_servers,
        value_serializer=lambda m: json.dumps(m).encode("utf-8"),
    )
    nlp = spacy.load("en_core_sci_sm")
    buffer = []
    for msg in consumer:
        buffer.append(msg.value)
        if len(buffer) >= cfg.stream_batch_size:
            df = pd.DataFrame(buffer)
            results = batch_predict_and_extract(
                model,
                tokenizer,
                df,
                cfg,
                nlp,
                batch_size=cfg.stream_batch_size,
                citation_analyzer=citation_analyzer,
            )
            for res in results:
                producer.send(cfg.kafka_output_topic, res)
            producer.flush()
            buffer = []


def main():
    cfg = Config()
    if cfg.label_names is None:
        cfg.label_names = ["Non-Cancer", "Cancer"]
    if cfg.candidate_models is None:
        cfg.candidate_models = {
            "PubMedBERT": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
            "BioBERT": "dmis-lab/biobert-base-cased-v1.1",
            "SciBERT": "allenai/scibert_scivocab_uncased",
            "ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
            "BioGPT": "microsoft/biogpt",
        }
    df = load_data(cfg.data_dir)
    df = preprocess(df)

    # convert integer labels to multi-hot vectors for multi-label training
    if cfg.problem_type == "multi_label_classification":
        df["label"] = df["label"].apply(lambda x: [1, 0] if x == 0 else [0, 1])

    nlp = spacy.load("en_core_sci_sm")
    df["diseases"] = df["abstract"].apply(lambda x: extract_diseases(nlp, x))

    train_df = df.sample(frac=0.8, random_state=42)
    eval_df = df.drop(train_df.index)

    citation_analyzer = CitationAnalyzer(cfg.citation_model_name)

    best_model = None
    best_tokenizer = None
    best_metrics = None
    best_name = None

    for name, ckpt in cfg.candidate_models.items():
        print(f"\nTraining {name} ({ckpt})...")
        cfg.model_name = ckpt
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        train_dataset = build_dataset(train_df, tokenizer, cfg.max_length)
        eval_dataset = build_dataset(eval_df, tokenizer, cfg.max_length)

        model, tok, metrics = train_baseline(cfg, train_dataset, eval_dataset)
        print(
            f"{name} Performance: Accuracy {metrics['accuracy']*100:.2f}% F1 {metrics['f1']:.2f}"
        )
        if best_metrics is None or metrics["f1"] > best_metrics["f1"]:
            best_model = model
            best_tokenizer = tok
            best_metrics = metrics
            best_name = name

    results = predict_and_extract(
        best_model,
        best_tokenizer,
        eval_df,
        cfg,
        nlp,
        citation_analyzer=citation_analyzer,
    )
    structured_path = structure_results(results, cfg.output_file)

    print(f"\nBest Model: {best_name}")
    print(f"Accuracy: {best_metrics['accuracy']*100:.2f}%")
    print(f"F1-score: {best_metrics['f1']:.2f}")
    print("Confusion Matrix:")
    print(best_metrics["confusion_matrix"])

    for res in results:
        print(json.dumps(res, indent=2))
    print(f"Structured results saved to {structured_path}")

    if cfg.stream:
        stream_papers(cfg, best_model, best_tokenizer, citation_analyzer)


if __name__ == "__main__":
    main()
