"""Autogen multi-agent workflow for the classification pipeline."""

from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import pipeline


def run_pipeline() -> None:
    """Run the research paper analysis pipeline using cooperating agents."""
    user = UserProxyAgent("user")

    data_agent = AssistantAgent(
        name="data_agent",
        system_message=(
            "Load and preprocess the dataset using pipeline.load_data and pipeline.preprocess. "
            "Store the processed DataFrame in shared memory under key 'df'."
        ),
    )

    train_agent = AssistantAgent(
        name="train_agent",
        system_message=(
            "Use the DataFrame from shared memory to train baseline and LoRA models "
            "with pipeline.train_baseline and pipeline.train_lora. "
            "Save models and metrics back to shared memory."
        ),
    )

    eval_agent = AssistantAgent(
        name="eval_agent",
        system_message=(
            "Run pipeline.predict_and_extract with the trained baseline model on the "
            "evaluation data. Output the path to the structured results JSON."
        ),
    )

    chat = GroupChat(agents=[user, data_agent, train_agent, eval_agent], messages=[])
    manager = GroupChatManager(groupchat=chat, llm_config={})

    user.initiate_chat(
        manager,
        message=(
            "Execute the full classification pipeline. "
            "Report training metrics and where the structured outputs are stored."
        ),
    )


if __name__ == "__main__":
    run_pipeline()
