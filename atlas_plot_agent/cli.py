import typer
import os
import yaml  # Ensure PyYAML is installed: `pip install pyyaml`
from agents import Agent, Runner


def load_secrets(secrets_file: str = "secrets.yaml"):
    """Load API secrets from a YAML file and set environment variables.

    Args:
        secrets_file (str): Path to the secrets file.
    """
    if os.path.exists(secrets_file):
        with open(secrets_file, "r") as file:
            secrets = yaml.safe_load(file)
            if "openai_api_key" in secrets:
                os.environ["OPENAI_API_KEY"] = secrets["openai_api_key"]
            else:
                raise KeyError("Missing 'openai_api_key' in secrets file.")
    else:
        raise FileNotFoundError(f"Secrets file '{secrets_file}' not found.")


def load_config(config_file: str = "agent-config.yaml"):
    """Load agent configuration from a YAML file.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        dict: The loaded configuration.
    """
    if os.path.exists(config_file):
        with open(config_file, "r") as file:
            return yaml.safe_load(file)
    else:
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")


def create_agents(agent_configs):
    """Create agents based on the provided configuration.

    Args:
        agent_configs (list): List of agent configurations.

    Returns:
        dict: A dictionary of agent instances keyed by their names.
    """
    agents = {}
    for agent_config in agent_configs:
        agent = Agent(
            name=agent_config["name"],
            instructions=agent_config["instructions"],
            model=agent_config["model"],
        )
        if "handoff_description" in agent_config:
            agent.handoff_description = agent_config["handoff_description"]
        agents[agent_config["name"]] = agent
    return agents


app = typer.Typer()


@app.command()
def run_agent(task: str, agent_name: str = "Orchestrator"):
    """Run a specific agent to perform a task.

    Args:
        task (str): What we need to do.
        agent_name (str): The name of the agent to use.
    """
    # Load secrets and configuration
    load_secrets("secrets.yaml")
    config = load_config("agent-config.yaml")
    agents = create_agents(config.get("agents", []))

    if agent_name not in agents:
        raise ValueError(f"Agent '{agent_name}' not found in configuration.")

    agent = agents[agent_name]

    # Process the task using the agent
    result = Runner.run_sync(agent, task)
    print(result.final_output)


if __name__ == "__main__":
    app()
