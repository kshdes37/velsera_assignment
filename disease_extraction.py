from pathlib import Path
import pandas as pd
import spacy
from tqdm import tqdm
from pipeline import load_data, preprocess, extract_diseases


def main(data_dir: Path = Path("Dataset"), model: str = "en_ner_bc5cdr_md", output_file: Path = Path("disease_mentions.csv")) -> None:
    """Extract diseases mentioned in each abstract and save the results."""
    df = load_data(data_dir)
    df = preprocess(df)

    nlp = spacy.load(model)

    diseases = []
    for text in tqdm(df["abstract"], desc="Extracting diseases"):
        diseases.append(extract_diseases(nlp, text))

    df["diseases"] = diseases
    df[["pmid", "label", "diseases"]].to_csv(output_file, index=False)
    print(f"Saved disease mentions to {output_file}")


if __name__ == "__main__":
    main()
