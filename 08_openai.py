from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
import webbrowser

client = OpenAI()

# list all models
models = client.models.list()
print(models.data[0].id)
print([mod.id for mod in models.data])

# # create our completion (using new API)
# completion = client.completions.create(model="gpt-3.5-turbo-instruct", prompt="Bill Gates is a")
# print(completion.choices[0].text)

# # generate images (using new API)
# image_gen = client.images.generate(
#     prompt="Zwei Hunde spielen unter einem Baum, cartoon",
#     n=2,
#     size="512x512"
# )
# # imgurl1 = image_gen.data[0].url
# # imgurl2 = image_gen.data[1].url
# # webbrowser.open(imgurl)
# for img in image_gen.data:
#     webbrowser.open_new_tab(img.url)


# # Gwendolyn Brooks Writers' Conference - Keynote Address: Dr. Donda West
# audio = open("audio/donda.mp3", "rb")
# transcript = client.audio.transcriptions.create(model="whisper-1", file=audio)
# print(transcript.text)
