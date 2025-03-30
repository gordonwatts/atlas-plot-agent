import asyncio
import typer
from agents import Agent, Runner, function_tool

from atlas_plot_agent.loader import (
    create_agents,  # Refactored imports
    load_config,
    load_secrets,
    load_tools,
)

app = typer.Typer()
ask_app = typer.Typer()
web_app = typer.Typer()

app.add_typer(ask_app, name="ask")
app.add_typer(web_app, name="web")


def initialize_agents():
    """Initialize and return agents from configuration."""
    load_secrets("secrets.yaml")
    config = load_config("agent-config.yaml")
    tools = load_tools(config.get("tools", []))
    return create_agents(config.get("agents", []), tools)


@ask_app.callback(invoke_without_command=True)
def ask(task: str, agent_name: str = "Orchestrator"):
    """Run a specific agent to perform a task.

    Args:
        task (str): What we need to do.
        agent_name (str): The name of the agent to use.
    """
    # Load secrets and configuration
    agents = initialize_agents()

    if agent_name not in agents:
        raise ValueError(f"Agent '{agent_name}' not found in configuration.")

    agent = agents[agent_name]

    # Process the task using the agent
    result = Runner.run_sync(agent, task)
    print(result.final_output)


@web_app.callback(invoke_without_command=True)
def web(agent_name: str = "Orchestrator"):
    """Run the web interface."""
    import streamlit as st

    @st.cache_data
    def cache_init_agents():
        return initialize_agents()

    agents = cache_init_agents()

    if agent_name not in agents:
        raise ValueError(f"Agent '{agent_name}' not found in configuration.")

    agent = agents[agent_name]

    # Deal with the fact we have an async library under us
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Set the title
    st.title("ATLAS Plot Agent")

    # If this is the first time we've run, we need to
    # setup our state across invocations.
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
        st.session_state.pending_input = []

    # Display conversation history.
    conversation_container = st.container()
    with conversation_container:
        for speaker, message in st.session_state.conversation:
            st.write(f"**{speaker}:** {message}")

    # Add the input text. DIsabled if we are working on
    # things.
    work_to_be_done = len(st.session_state.pending_input) != 0
    if work_to_be_done:
        st.session_state.user_input = ""
    user_input = st.text_input(
        "You:",
        key="user_input",
        label_visibility="collapsed",
        disabled=work_to_be_done,
        value="",
        placeholder="Enter your query...",
    )

    # Lets see if they are ready for us to do some sort of chat here!
    if st.button("Send", disabled=work_to_be_done):
        if user_input:
            st.session_state.conversation.append(("You", user_input))
            st.session_state.pending_input.append(user_input)
            with conversation_container:
                st.write(f"**You:** {user_input}")

            # We can just go re-run the script at this point.
            st.rerun()

    # Run the LLM here... this hangs all input, unfortunately...
    # but for now...
    if work_to_be_done:
        input = st.session_state.pending_input.pop()

        response = Runner.run_sync(agent, input)
        st.session_state.conversation.append(("Agent", response.final_output))

        st.rerun()


if __name__ == "__main__":
    app()


@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is cloudy."
