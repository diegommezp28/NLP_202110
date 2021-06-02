from deep_translator import GoogleTranslator

# %%
import json
import os
import yaml
import pickle5 as pk
import pandas as pd


def pickleToJson(path):
    df = None
    with open(path, "rb") as output_file:
        df = pk.load(output_file)
    jsonFileName = path.replace(".pkl", ".json")
    df.to_json(jsonFileName, orient="records", lines=True)


#%%
with open("./configIter2.yaml", "r+") as file:
    config = yaml.safe_load(file)
    filename = config["filename"]
    inputFilePickle = f"{filename}.pkl"
    inputFile = f"{filename}.json"
    outputFile = f"{filename}_tagged.json"
    if not os.path.isfile(inputFile):
        print("First time creating the files")
        pickleToJson(inputFilePickle)
    translate = config["translate"]
    tags = config["tags"]
    source_lang = config["source_lang"]
    target_lang = config["target_lang"]

    def cls():
        os.system("cls" if os.name == "nt" else "clear")

    def tagInput():
        tag = input(
            "\nPlease select a tag:"
        )  # \nOr type 'save' to save your progress")
        try:
            index = int(tag)
            if index not in range(0, len(tags)):
                print("Input not in range, please try again")
                return tagInput()
            return tags[int(tag)]
        except:
            print("Invalid input, please try again")
            return tagInput()

    with open(inputFile, "r") as inf:
        with open(outputFile, "a") as outf:
            for i, line in enumerate(inf):
                if i <= config["current_index"]:
                    continue
                json_line = json.loads(line)

                print("Tags:")
                for index, tag in enumerate(tags):
                    print(f"• {tag} [{index}]")

                print("\nGenerated Tag:", json_line["simpleTag"])

                print("\nOriginal:")
                print(json_line["text"])
                if translate:
                    translated = GoogleTranslator(source="auto", target="es").translate(
                        json_line["text"]
                    )
                    print()
                    print("Translated:")
                    print(translated)

                json_line["tag"] = tagInput()
                print("after")
                json.dump(json_line, outf)
                outf.write("\n")
                config["current_index"] = i
                file.seek(0)
                file.truncate()
                yaml.dump(config, file)
                cls()
