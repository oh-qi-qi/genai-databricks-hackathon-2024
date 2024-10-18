from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

def print_nested_dict_display(data):
    console = Console()
    def format_value(value):
        if isinstance(value, str):
            try:
                json_data = json.loads(value)
                return Markdown(f"```json\n{json.dumps(json_data, indent=2)}\n```")
            except json.JSONDecodeError:
                return Markdown(value)
        elif isinstance(value, dict):
            return Markdown(f"```json\n{json.dumps(value, indent=2)}\n```")
        else:
            return str(value)

    for key, value in data.items():
        formatted_value = format_value(value)
        panel = Panel(formatted_value, title=key, expand=False)
        console.print(panel)