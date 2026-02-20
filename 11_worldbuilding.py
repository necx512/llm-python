import os
import io
from PIL import Image
from dotenv import load_dotenv
import pandas as pd
import cohere
from stability_sdk import client


load_dotenv()

co = cohere.Client(os.getenv("COHERE_API_KEY"))
stability = client.StabilityInference(
    key=os.getenv("STABILITY_API_KEY"),
    verbose=True
)

# predicted = co.generate(prompt="Civilization is a video game", 
#                         num_generations=5,
#                         temperature=0.8,
#                         return_likelihoods="GENERATION", #ALL, NONE
#                         max_tokens=80
#                         )

# print(predicted)

storyConfig = {
    "titlePrompt": """Realistic video game title for a game inspired by Civilization, Starbound and Surviving Mars. Turn-based, deep tech trees, single player modes only with card-based mechanics. Examples: Colonizing Mars, Space Empires, Interstellar Frontiers. Generate one title.""",
    "civLeadersPrompt": """Create a fictional leader for a space-themed Civilization video game. Format: Leader: [Name] | Civilization: [Civ Name] | Description: [brief description]""",
    "characterStyle": "in-game character portraits, sci-fi, futuristic civilization in the background, serious expression, realistic",
    "civLocationPrompt": """Create a fictional civilization for a space-themed video game. Format: Civilization: [Name] | Description: [brief description]""",
    "civStyle": "realistic in-game space civilization cities and space ports, thriving, busy, sci-fi, hi-resolution scenery for a city simulation game",
    "inGameCutScenes": """
        Continue writing the in-game cut scenes following the format of the dialog provided below:

        Welcome, Commander. You have been appointed as the new leader of the {civ} civilization.
        {civ_description}.
        Your mission is to lead your people to prosperity and glory among the stars. 

        You will need to build a thriving civilization, explore the galaxy, and defend your people from hostile elements. Central Officer Johann Bradford will be at your aid. Our chief scientist, Dr. Maya will also assist you in your quest for galactic domination.
        Both of them are waiting for you in the command center. Please proceed to the command center to begin your daily briefing. 

        Bradford: Welcome, Commander. I am Central Officer Johann Bradford. You came in at a good time. We have just received a distress signal from the {civ2} civilization. Their nearby starport, the Mercury Expanse, has alerted us to some hostile activity in their vicinity.
        Dr.Maya: I urge caution, Commander. {civ2_description}. In their past encounters with {ally1}, they have proven to be distrustful and quick to escalate conflicts. I recommend that we send a small fleet to investigate the situation first. 
        Bradford: Perhaps Military Officer Levy can lead the investigation. Our senior-ranked officers are due to be back from their planetary exploration mission soon. 

        Player: [Select Military Officer Levy to lead the investigation] or [Wait for the senior-ranked officers to return from their planetary exploration mission] 

        Bradford: Commander, we have just received another distress signal from the {civ2} civilization on behalf of Mercury Expanse. They are reporting heavy casualties on the ground and will require immediate assistance.
        Dr.Maya: Commander, I strongly recommend we send a small fleet to investigate the situation. Our senior-ranked officers are away on their planetary expeditions and we cannot afford to take any chances.
        Bradford: I disagree. Mercury Expanse is an economically important starport and {civ2} is a valuable trading partner. Sending an investigation fleet while their star system is under attack will only signal distrust and hostility. They are reporting heavy casualties.
        Dr.Maya: I understand your concerns, Commander. But we cannot afford to take any chances until our Team 6 and Delta Squad return from their expeditions. If things go wrong, we will be left with no viable defense options.

        Player: [Summon Team 6 to return from their expeditions] or [Summon Delta Squad to return from their expeditions] \n

        Bradford: While we wait for our expedition team to return, do I have orders to send our Delta Reserve to assist the {civ2} civilization?
        Dr.Maya: Commander...

        Bradford: Commander, Officer Levy is on the line. Shall I put him through?

        Player: Yes.
        Officer Levy: Commander, I've heard about the situation at Mercury Expanse. I'm concerned about the severity on the ground if the reports are accurate. A small fleet of investigation ship is no match for a full-scale attack.
        Dr.Maya: Being economically important, Mercury Expanse has a large fleet of defense ships and ground troops. I wonder what could have caused such heavy casualties on the ground, and if so, Officer Levy is right. We will need to send in our entire reserve fleet to have the best chance of success. This puts a lot of risk on our side, but it might be the only way to ensure the safety of {civ2} and Mercury Expanse.

        Player: [Send in the entire reserve fleet] or [Send the Delta Reserve while preserving the rest of the fleet] \n
        Bradford:
    """
}


# ====================================================================================
# MIGRATED FROM GENERATE API TO CHAT API (Generate API removed September 15, 2025)
# ====================================================================================
# Original code used co.generate() which supported:
# - Multiple generations via num_generations parameter
# - Token-level likelihood scores via return_likelihoods="GENERATION"
#
# Chat API changes:
# - No num_generations parameter (must call multiple times)
# - No token likelihoods (removed likelihood scoring)
# - Uses conversational chat format
#
# KNOWN LIMITATIONS:
# - Chat API output format differs from Generate API
# - May not follow exact prompt templates (e.g., "Name | Description: Text")
# - Script may encounter parsing errors with certain outputs
# - Requires prompt engineering to match original output format
# ====================================================================================

def generate(prompt, model="command-a-03-2025", num_generations=5, temperature=0.7, max_tokens=64):
    """
    Migrated from Cohere Generate API to Chat API.
    Note: Token likelihood scoring removed as it's not available in Chat API.
    Model updated to "command-a-03-2025" (most performant model as of 2026).
    """
    generations = {}

    # Chat API doesn't support num_generations, so call multiple times
    for i in range(num_generations):
        response = co.chat(
            message=prompt,
            model=model,  # Using command-a-03-2025 (most performant Chat API model)
            temperature=temperature,
            max_tokens=max_tokens
            # Note: Removed stop_sequences - Chat API handles differently than Generate API
        )

        # Take only first line to match original behavior
        text = response.text.split("\n")[0].strip()

        # Chat API doesn't provide token likelihoods
        # Use generation order as proxy (earlier = higher "likelihood")
        generations[text] = num_generations - i

    # turn this into a dataframe (sorted by generation order)
    df = pd.DataFrame.from_dict(generations, orient="index", columns=["likelihood"])
    df = df.sort_values(by=["likelihood"], ascending=False)
    return df

def generate_img(prompt):
    predictions = stability.generate(prompt)

    for img in predictions:
        for artifact in img.artifacts:
            # check that it's an image type
            if artifact.type == 1:
                img = Image.open(io.BytesIO(artifact.binary))
                img.show()
                return img


if __name__ == "__main__":
    titles = generate(storyConfig["titlePrompt"])
    print(titles)

    leaders = generate(storyConfig["civLeadersPrompt"], num_generations=5)
    print(leaders)

    # leader_img = generate_img(
    #     f"{storyConfig['characterStyle']}{leaders.iloc[3].name}"
    # )
    # leader_img.show()

    civs = generate(storyConfig["civLocationPrompt"])

    # for civ in civs[:3]:
    #     civName, civDescription = civs.name.split("|")
    #     civImg = generate_img(f"{storyConfig['civStyle']}{civDescription}")
    #     civImg.show()

    try:
        civname, civdescription = civs.iloc[0].name.split("| Description: ")
    except:
        civname, civdescription = "Unknown Civilization", "A mysterious space civilization"

    try:
        civname2, civdescription2 = civs.iloc[1].name.split("| Description: ")
    except:
        civname2, civdescription2 = "Distant Civilization", "A distant space civilization"

    try:
        ally1 = leaders.iloc[0].name.split("|")[0].strip()
    except:
        ally1 = "Unknown Leader"

    cutscenes_prompt = storyConfig["inGameCutScenes"].format(
        civ=civname,
        civ_description=civdescription,
        civ2=civname2,
        civ2_description=civdescription2,
        ally1=ally1
    )

    cutscenes = generate(
        cutscenes_prompt,
        model="command-a-03-2025",  # Using most performant Cohere Chat API model
        num_generations=3,
        max_tokens=800,
        temperature=0.9,
    )

    dialog = cutscenes_prompt + "\n" + cutscenes.iloc[0].name
    print(dialog)

    # generate img for that cutscene
    cutscene_img = generate_img(
        f"An in-game, sci-fi cutscene with lots of details for: {cutscenes.iloc[0].name}"
    )
