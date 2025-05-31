import json
from pathlib import Path
import pandas as pd
from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent


def load_results(path: Path) -> pd.DataFrame:
    """Load structured results JSON into a DataFrame."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return pd.json_normalize(data)


def main(results_file: Path = Path("outputs/structured_outputs.json")) -> None:
    """Interactive query engine powered by LangChain."""
    df = load_results(results_file)
    llm = OpenAI(temperature=0)
    agent = create_pandas_dataframe_agent(llm, df, verbose=False)
    print("Enter a question about the results (type 'exit' to quit):")
    while True:
        question = input("> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        answer = agent.run(question)
        print(answer)


if __name__ == "__main__":
    main()
