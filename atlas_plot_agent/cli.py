import typer
import os
import yaml  # Ensure PyYAML is installed: `pip install pyyaml`
from agents import Agent, Runner


def load_config(config_file: str = ".agent_config"):
    """Load configuration from a YAML file and set environment variables.

    Args:
        config_file (str): Path to the configuration file.
    """
    if os.path.exists(config_file):
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
            if "openai_api_key" in config:
                os.environ["OPENAI_API_KEY"] = config["openai_api_key"]
            else:
                raise KeyError("Missing 'openai_api_key' in configuration file.")
    else:
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")


# Load configuration
load_config("atlas-agent.yaml")

app = typer.Typer()

# Define the local tool


def verify_dataset(dataset_path: str) -> str:
    """Verify the dataset at the given path.

    Args:
        dataset_path (str): Path to the dataset.

    Returns:
        str: Verification result.
    """
    # Placeholder logic for dataset verification
    return f"Dataset at {dataset_path} has been verified."


@app.command()
def run_agent(task: str):
    """Run everything.

    Args:
        task (str): What we need to do
    """
    # Create the top level tool
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant",
        model="gpt-4o-mini",
    )

    # Process the task using the agent

    result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
    print(result.final_output)


if __name__ == "__main__":
    app()
