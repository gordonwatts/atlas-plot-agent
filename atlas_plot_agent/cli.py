import typer
from agents import Agent, Runner, function_tool
from atlas_plot_agent.loader import (
    load_secrets,
    load_config,
    create_agents,
    load_tools,
)  # Refactored imports

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
    tools = load_tools(config.get("tools", []))
    agents = create_agents(config.get("agents", []), tools)

    if agent_name not in agents:
        raise ValueError(f"Agent '{agent_name}' not found in configuration.")

    agent = agents[agent_name]

    # Process the task using the agent
    result = Runner.run_sync(agent, task)
    print(result.final_output)


if __name__ == "__main__":
    app()


@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is cloudy."
