# ====================================================================================
# MIGRATED FROM GUIDANCE 0.x TO 1.x (Major API changes)
# ====================================================================================
# Original code used Handlebars-style templates ({{#system}}, {{gen}}, etc.)
# New guidance 1.x uses Python decorators and function-based API
# ====================================================================================

from dotenv import load_dotenv
from guidance import models, gen, system, user, assistant

load_dotenv()

# Initialize model (guidance 1.x syntax)
llm = models.OpenAI("gpt-3.5-turbo")

# Define the OS to teach
os_name = "Linux"

# Build the conversation using guidance 1.x API
with system():
    lm = llm + f"You are a CS Professor teaching {os_name} systems administration to your students."

with user():
    lm += f"What are some of the most common commands used in the {os_name} operating system? Provide a one-liner description. List the commands and their descriptions one per line. Number them starting from 1."

with assistant():
    lm += gen("commands", max_tokens=100)

with user():
    lm += "Which among these commands are beginners most likely to get wrong? Explain why the command might be confusing. Show example code to illustrate your point."

with assistant():
    lm += gen("confusing_commands", max_tokens=100)

# Extract results
print(lm["commands"])
print("===")
print(lm["confusing_commands"])
