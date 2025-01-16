import numpy as np
import openai
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from nn import forward, train_X, train_Y

file_path = 'input_transformed.xlsx'
excel_data = pd.ExcelFile(file_path)
sheet_names = excel_data.sheet_names
changes_df = excel_data.parse("Changes")
transformations_df = excel_data.parse("Transformations")
all_attributes = set(transformations_df.columns)
attributes_in_changes = set(changes_df["Column"].unique())
file_descriptions = 'descriptions.txt'
predictions_file = 'predictions.txt'

def load_weights_from_text(file_name):
    weights = []
    biases = []

    with open(file_name, "r") as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("Weights Layer"):
                i += 1
                weight_matrix = []
                while i < len(lines) and not lines[i].startswith("Biases Layer"):
                    weight_matrix.append(list(map(float, lines[i].strip().split())))
                    i += 1
                weights.append(np.array(weight_matrix))
            if i < len(lines) and lines[i].startswith("Biases Layer"):
                i += 1
                bias_vector = []
                while i < len(lines) and lines[i].strip():
                    bias_vector.append(list(map(float, lines[i].strip().split())))
                    i += 1
                biases.append(np.array(bias_vector).flatten())
            i += 1

    return weights, biases

openai.api_key = "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"

mappings_dict = {}
for _, row in changes_df.iterrows():
    column = row["Column"]
    if column.lower() == "race":
        continue  # skip  'Race'
    if column.lower() in ['sex','age','cats in the household','stays in','lives in']:
        continue
    original_value = row["Original Value"].lower()
    new_value = row["New Value"]
    if column not in mappings_dict:
        mappings_dict[column] = {}
    mappings_dict[column][original_value] = new_value
print(mappings_dict)
#mapping from word to numbers for each column


def extract_attributes_from_description(description):
    mapping_info = "Here are the mappings for each attribute:\n"
    for column, mappings in mappings_dict.items():
        mapping_info += f"{column}: {mappings}\n"

    prompt = (
        f"Extract attributes from the following description:\n"
        f"{description}\n"
        f"Use the following mappings for reference:\n"
        f"{mapping_info}"
        f"The output must be the form attribute:value (not the number) "
        f"Attribute must be the same as in the mapping reference. The value extracted must be the same as in the mapping reference! "
        f"You must keep the word considerable and moderate not any other form (such as considerably or moderately)."
        f"The attributes can vary a bit like from calm, you can have calmness or from timid timidity and so on for the other attributes."
        f"Even if the mapping values does not find exactly in the description, you should output the initial mapping values, not the values you found in the description.")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant trained to extract attributes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )

        content = response['choices'][0]['message']['content']
        print(f"GPT response content: {content}")

        with open(file_descriptions, 'a') as file:
            file.write("\n" + description + "\n")

        with open(file_descriptions, 'a') as file:
            file.write(content + "\n")

        return content
    except Exception as e:
        print(f"Error: {e}")
        return {}
#use gpt3.5
#return column: value in string

def get_input_nn(out_gpt):
    input_dict = {}
    for line in out_gpt.strip().split('\n'):
        key, value = line.split(':', 1)
        input_dict[key.strip().lower()] = value.strip().lower()

    output_dict = {}
    for key in mappings_dict.keys():
        value = input_dict.get(key.lower(), None)
        if value in mappings_dict[key]:
            output_dict[key] = mappings_dict[key][value]
        else:
            output_dict[key] = 0

    list_values = []
    for key in output_dict.keys():
        list_values.append(output_dict[key])
    #print(output_dict)
    # mapping columns-numbers from text

    return output_dict, list_values
#returns vector of numbers for nn

def get_breed_from_text(out_from_gpt):
    _, value_numbers_for_nn = get_input_nn(out_from_gpt)
    #value_numbers_for_nn = [2, 2, 3, 3, 1, 1, 3, 4, 1, 3, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 4, 1, 2, 1, 1]

    x_min = train_X.min(axis=0).to_numpy()
    x_max = train_X.max(axis=0).to_numpy()

    normalized_input = (value_numbers_for_nn - x_min) / (x_max - x_min)
    normalized_input = normalized_input.reshape(1, -1)

    weights_file = "neural_network_weights.txt"
    weights, biases = load_weights_from_text(weights_file)
    W1, W2, W3 = weights
    b1, b2, b3 = biases

    _, _, _, _, _, prediction = forward(normalized_input, W1, b1, W2, b2, W3, b3, training=False)
    predicted_class = np.argmax(prediction, axis=1)[0]

    breed_mapping = changes_df[changes_df["Column"] == "Race"].set_index("New Value")["Original Value"].to_dict()
    predicted_breed = breed_mapping.get(predicted_class + 1, "Unknown")
    print(f"Predicted breed: {predicted_breed}")
    return predicted_breed

def read_and_identify():
    description = ("This female cat is aged between 2 and 10 years. She lives in a rural area and shares her household with two other cats, making a total of 3 cats in the household.She stays in an apartment without a balcony and does not spend any time outdoors each day. She also does not spend any time with her owner.Her personality traits include being moderate in timidity, a bit calm, and moderate in fear. She is extremely intelligent, a bit vigilant, and extremely persevering. While she is only a bit loving and moderate in amicability, she experiences moderate loneliness. She is not brutal, not aggressive, and not impulsive.However, she is considerable in dominance and extremely predictable while being considerably distracted.Regarding her environment, she lives in a high natural area and never captures birds or small mammals."
)
    description =("This female cat, aged between 2 and 10 years, resides in a rural area within an apartment without a balcony. She does not spend any time outdoors each day and does not have interaction with her owner. Her temperament is characterized by considerable timidity, a slight level of calmness, and a moderate level of fear. Despite her reserved nature, she exhibits an extremely high level of intelligence and perseverance. In terms of social behavior, she is extremely loving but only a bit amicable, experiencing moderate loneliness and a slight sense of dominance. She is not aggressive, impulsive, or predictable, and only slightly vigilant and distracted. This cat has a high affinity for natural areas but never engages in capturing birds or small mammals.")

    description = ("The cat is considerable calmness and perseverance, making it a good companion with a moderate level of vigilance. It is not afraid, aggressive, or lonely, and has a considerable level of amiability and predictability.")

    description = ("The  cat is known for its considerable calmness and amiability towards its owner, making it a good companion. It shows a moderate level of vigilance and perseverance, while also displaying considerable predictability and impulsiveness. This breed tends to have a considerable dominance and a moderate brutal nature, although it is not aggressive. The  cat enjoys spending time outdoors in a moderate natural area but rarely captures birds and never captures small mammals.")

    #description = ("On the other hand, the cat spends long hours outdoors but has limited interaction with its owner. It is a bit timid and calm, not afraid, and extremely intelligent. While it is only a bit vigilant, it is considerably persevering and loving. The cat is a bit amical, not lonely, and extremely brutal, but not dominant or aggressive. It is a bit impulsive and predictable, with considerable distractions. This cat has a high natural area and never captures birds, rarely capturing small mammals.")

    description = ("The cat is considerable calm and perseverance. The cat is not aggressive and extremely amical")

    description = ("The cat is not calm and not perseverance. The cat is extremely aggressive and not amical.")


    extracted_attributes = extract_attributes_from_description(description)
    predicted_race = get_breed_from_text(extracted_attributes)
    with open(predictions_file, 'a') as file:
        file.write("\n" + description + "\n")
    with open(predictions_file, 'a') as file:
        file.write(str(extracted_attributes) + "\n")
    with open(predictions_file, 'a') as file:
        file.write(predicted_race + "\n")
read_and_identify()

