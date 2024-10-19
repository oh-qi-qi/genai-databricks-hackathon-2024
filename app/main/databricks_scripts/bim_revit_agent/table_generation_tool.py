# Stadard library imports
import os
import re
import ast
import json
import logging
from typing import List, Dict, Any, Union

# Third-party imports
import numpy as np
import pandas as pd

class TableGenerationTool:
    def __init__(self):
        self.max_rows = 10
        self.max_columns_to_show = 10

    def generate_markdown_table(self, input_data: str) -> str:
        """
        Generate a Markdown table from input JSON data.
        """
        try:
            data = self._parse_input(input_data)
            if not data:
                return "Error: Invalid or empty data"

            total_items = len(data)
            headers = self._get_headers(data)

            # Extract first 10 rows and last row if data is large
            if total_items > self.max_rows + 1:
                displayed_data = data[:self.max_rows] + [data[-1]]
                ellipsis_needed = True
            else:
                displayed_data = data
                ellipsis_needed = False

            table = self._create_table_header(headers)
            table += self._create_table_rows(displayed_data, headers, ellipsis_needed)
            table += self._add_table_footer(total_items, len(headers))

            return table
        except json.JSONDecodeError as e:
            return f"Error parsing JSON: {str(e)}\nInput data: {input_data[:100]}..."
        except Exception as e:
            return f"Error generating table: {str(e)}\nInput data: {input_data[:100]}..."

    def _parse_input(self, input_data: str) -> List[Dict[str, Any]]:
        """Parse the input string as JSON."""
        return json.loads(input_data)

    def _get_headers(self, data: List[Dict[str, Any]]) -> List[str]:
        """Extract unique headers from all items in the data."""
        headers = set()
        for item in data:
            headers.update(item.keys())
        return sorted(list(headers))

    def _create_table_header(self, headers: List[str]) -> str:
        """Create the Markdown table header."""
        visible_headers = headers[:self.max_columns_to_show]
        if len(headers) > self.max_columns_to_show:
            visible_headers.append("...")
        header_row = "| " + " | ".join([inflection.titleize(header.replace("_", " ").replace("source", "Start").replace("target", "Destination")) for header in visible_headers]) + " |\n"

        separator_row = "|" + "|".join(["---" for _ in visible_headers]) + "|\n"
        return header_row + separator_row

    def _create_table_rows(self, data: List[Dict[str, Any]], headers: List[str], ellipsis_needed: bool) -> str:
        """Create the Markdown table rows."""
        rows = ""
        total_rows = len(data)
        for idx, item in enumerate(data):
            if ellipsis_needed and idx == self.max_rows:
                # Insert ellipsis row
                row_data = ["..." for _ in headers[:self.max_columns_to_show]]
                if len(headers) > self.max_columns_to_show:
                    row_data.append("...")
                row = "| " + " | ".join(row_data) + " |\n"
                rows += row
                continue
            row_data = [str(item.get(header, "")) for header in headers[:self.max_columns_to_show]]
            if len(headers) > self.max_columns_to_show:
                row_data.append("...")
            row = "| " + " | ".join(row_data) + " |\n"
            rows += row
        return rows

    def _add_table_footer(self, total_items: int, total_columns: int) -> str:
        """Add a footer with information about the number of items and columns."""
        footer = f"\n*Table {'truncated' if total_items > self.max_rows + 1 else 'complete'}. "
        footer += f"Showing {min(self.max_rows + 1, total_items)} out of {total_items} total records. "
        if total_columns > self.max_columns_to_show:
            footer += f"Displaying {self.max_columns_to_show} out of {total_columns} columns.*"
        else:
            footer += f"All {total_columns} columns displayed.*"
        return footer
