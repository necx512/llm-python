# ====================================================================================
# MIGRATED FROM GUIDANCE 0.x TO 1.x (Major API changes)
# ====================================================================================
# Original code used advanced Handlebars features ({{#geneach}}, {{select}}, etc.)
# New guidance 1.x uses Python loops and simplified API
# Some features simplified due to API changes
# ====================================================================================

import random
from dotenv import load_dotenv
from guidance import models, gen, user, assistant

load_dotenv()

# Initialize model (guidance 1.x syntax)
llm = models.OpenAI("gpt-3.5-turbo")  # Updated from text-davinci-003

os_name = "Linux"

# Select quiz flavor
quizflavor = [
    "Quiz of the day!",
    "Test your knowledge!",
    "Here is a quiz!",
    "You think you know Unix?",
]
flavor = random.choice(quizflavor)
randomPts = random.randint(1, 5)

# Build the prompt (all text must be within role contexts)
with user():
    lm = llm + f"What are the top ten most common commands used in the {os_name} operating system? Provide a one-liner description for each command. List them numbered from 1-10."

with assistant():
    lm += gen("commands_list", max_tokens=300)

with user():
    lm += f"\n{flavor}\nExplain the following commands for 🥇 {randomPts} points (pick 3 from the list above):"

with assistant():
    lm += gen("quiz_explanation", max_tokens=200)

# Print results
print("Generated commands:")
print(lm["commands_list"])
print("\n===")
print("Quiz explanation:")
print(lm["quiz_explanation"])
