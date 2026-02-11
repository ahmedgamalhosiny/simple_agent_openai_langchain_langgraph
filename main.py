from typing import List
import json
import random
import string
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import gradio as gr

from dotenv import load_dotenv
load_dotenv()

# ----------------------tools---------------------------

@tool
def write_json(filepath: str, data: dict) -> str:
    """Write Python dictionary as JSON to a file with pretty fromatting."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return f"Successfully wrote JSON data to '{filepath}' ({len(json.dump(data))} characters)."
    except Exception as e:
        return f"Error writing JSON: {str(e)}"
    

@tool
def read_json(file_path: str) -> str:
    """Read a Python dictionary as JSON to a file with pretty fromatting."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
    except FileNotFoundError:
        return f"Error: file '{file_path}' not found."
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON in file - {str(e)}"
    except Exception as e:
        return f"Error reading JSON: {str(e)}"
    


@tool
def generate_simple_users(
                        first_names: List[str],
                        last_names: List[str],
                        domains: List[str],
                        min_age: int,
                        max_age: int
                        ) -> dict:
    """
    Generate sample user data. Count is determined by the length of the first_names.

    Args:
        first_names: List of first names (one per user)
        last_names: List of last names (will cycle if fewer than first_names)
        last_names: List of domains (will cycle through)
        min_age: Minimum age for users
        max_age: Minimum age for users

    Returns:
        Dictionary with 'users' array or 'error' message
    """
    # validation
    if not first_names:
        return {"error": "first names list cannot be empty"}
    if not last_names:
        return {"error": "last names list cannot be empty"}
    if not domains:
        return {"error": "domains list cannot be empty"}
    if min_age > max_age:
        return {"error": f"min_age ({min_age}) cannot be greater than ({max_age})"}
    if min_age < 0 or max_age < 0:
        return {"error": "ages must be non-negative"}

    users = []
    count = len(first_names)

    for i in range(count):
        first = first_names[i]
        last = last_names[i % len(last_names)]
        domain = domains[i % len(domains)]
        email = f"{first.lower()}.{last.lower()}@{domain}"

        user = {
            "id": i + 1,
            "firstName": first,
            "lastName": last,
            "email": email,
            "username": f"{first.lower()}{random.randint(100, 999)}",
            "age": random.randint(min_age, max_age),
            "registeredAt": (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat()
            }
        
        users.append(user)

    return {"users": users, "count": len(users)}



TOOLS = [write_json, read_json, generate_simple_users]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

SYSTEM_MESSAGE = (
    "You are DataGen, a helpful assistant that generates sample data for applications. "
    "To generate users, you need: first_names (list), last_names (list), domains (list), min_age, max_age. "
    "When asked to save users, first generate them with the tool, then immediately use write_json with the results. "
    "If the user refers to 'those users' from a previous request, ask them to specify the details again. "
)

agent = create_react_agent(llm, TOOLS, prompt=SYSTEM_MESSAGE)

def run_agent(user_input: str, history: List[BaseMessage]) -> AIMessage:
    """Singel-turn agent runner with automatic tool execution via LangGraph."""
    try:
        result = agent.invoke(
            {"message": history + [HumanMessage(content=user_input)]},
            config={"recursion_limit": 50}
        )

        return result["messages"][-1]

    except Exception as e:

        return AIMessage(content=f"Error: {str(e)}\n\nPlease try rephrasing your request or provide more specific details")
    


if __name__ == "__main__":

    def chat_function(message, history):
        # Convert Gradio history to List[BaseMessage]
        messages = []
        for user_msg, bot_msg in history:
            messages.append(HumanMessage(content=user_msg))
            if bot_msg:
                messages.append(AIMessage(content=bot_msg))

        # Add current message
        messages.append(HumanMessage(content=message))

        # Run agent with current message and previous history
        response = run_agent(message, messages)

        # Return updated history
        return history + [(message, response.content)]

    def example_click(example_text):
        return example_text

    with gr.Blocks() as demo:
        gr.Markdown("# DataGen Agent\n**Generate sample user data and save to JSON files**")

        gr.Markdown("**How to use:** Simply type your request in natural language. The agent will generate user data and save it to JSON files automatically.")

        gr.Markdown("**Quick Examples:**")

        with gr.Row():
            example1 = gr.Button("Generate users named John, Jane, Mike and save to users.json", size="sm")
            example2 = gr.Button("Create users with last names Smith, Jones", size="sm")
            example3 = gr.Button("Make users aged 25-35 with company.com emails", size="sm")

        chatbot = gr.Chatbot(height=500, show_label=False)
        msg = gr.Textbox(
            placeholder="Type your request here... (e.g., 'Generate 5 users with names Alice, Bob, Charlie')",
            label="Your Message",
            lines=1,
            show_label=False
        )
        with gr.Row():
            submit_btn = gr.Button("🚀 Send", variant="primary", size="sm")
            clear = gr.ClearButton([msg, chatbot], value="🗑️ Clear Chat", size="sm")

        gr.Markdown("---\n*Powered by LangChain & OpenAI*")

        # Event handlers
        msg.submit(chat_function, [msg, chatbot], [chatbot])
        submit_btn.click(chat_function, [msg, chatbot], [chatbot])
        example1.click(example_click, inputs=[example1], outputs=[msg])
        example2.click(example_click, inputs=[example2], outputs=[msg])
        example3.click(example_click, inputs=[example3], outputs=[msg])

    demo.launch(theme=gr.themes.Soft())
