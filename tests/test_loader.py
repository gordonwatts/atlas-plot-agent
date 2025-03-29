import os
import unittest
from unittest.mock import patch, mock_open, MagicMock
from atlas_plot_agent.loader import load_secrets, load_config, create_agents, load_tools
from agents import Agent


class TestLoader(unittest.TestCase):
    @patch("os.path.exists", return_value=True)
    @patch(
        "builtins.open", new_callable=mock_open, read_data="openai_api_key: test_key"
    )
    def test_load_secrets(self, mock_file, mock_exists):
        load_secrets("secrets.yaml")
        self.assertEqual(os.environ["OPENAI_API_KEY"], "test_key")

    @patch("os.path.exists", return_value=False)
    def test_load_secrets_file_not_found(self, mock_exists):
        with self.assertRaises(FileNotFoundError):
            load_secrets("missing_secrets.yaml")

    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data="{}")
    def test_load_secrets_missing_key(self, mock_file, mock_exists):
        with self.assertRaises(KeyError):
            load_secrets("secrets.yaml")

    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data="key: value")
    def test_load_config(self, mock_file, mock_exists):
        config = load_config("agent-config.yaml")
        self.assertEqual(config, {"key": "value"})

    @patch("os.path.exists", return_value=False)
    def test_load_config_file_not_found(self, mock_exists):
        with self.assertRaises(FileNotFoundError):
            load_config("missing_config.yaml")

    def test_create_agents(self):
        agent_configs = [
            {
                "name": "Agent1",
                "instructions": "Do something",
                "model": "gpt-3.5",
                "tools": ["Tool1"],
                "handoffs": ["Agent2"],
            },
            {
                "name": "Agent2",
                "instructions": "Do something else",
                "model": "gpt-3.5",
            },
        ]
        tools = {"Tool1": MagicMock()}
        agents = create_agents(agent_configs, tools)

        self.assertIn("Agent1", agents)
        self.assertIn("Agent2", agents)
        self.assertEqual(agents["Agent1"].tools, [tools["Tool1"]])
        self.assertEqual(agents["Agent1"].handoffs, [agents["Agent2"]])

    def test_create_agents_empty_list(self):
        with self.assertRaises(ValueError):
            create_agents([], {})

    def test_create_agents_missing_tool(self):
        agent_configs = [
            {
                "name": "Agent1",
                "instructions": "Do something",
                "model": "gpt-3.5",
                "tools": ["Tool1"],
            }
        ]
        tools = {}
        with self.assertRaises(ValueError):
            create_agents(agent_configs, tools)

    def test_load_tools(self):
        tool_configs = [
            {"name": "Tool1", "type": "module.func"},
            {"name": "Tool2", "type": "module.another_func"},
        ]
        with patch("importlib.import_module") as mock_import:
            mock_import.return_value = MagicMock(
                func=MagicMock(), another_func=MagicMock()
            )
            tools = load_tools(tool_configs)

        self.assertIn("Tool1", tools)
        self.assertIn("Tool2", tools)

    def test_load_tools_empty_list(self):
        tools = load_tools([])
        self.assertEqual(tools, {})


if __name__ == "__main__":
    unittest.main()
