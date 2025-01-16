import openai
from identify_gpt import load_weights_from_text, changes_df, mappings_dict
from nn import input_size, relu_derivative, forward, train_X
import numpy as np

def denormalize_and_clip(normalized_data):
    normalized_data = np.array(normalized_data)
    x_min_np = np.array(x_min)
    x_max_np = np.array(x_max)
    #print(f"x_min_n={x_max_n}")

    denormalized_data = normalized_data * (x_max_np - x_min_np) + x_min_np

    denormalized_data = np.round(denormalized_data).astype(int)
    denormalized_data = np.clip(denormalized_data, x_min_np, x_max_np)

    return denormalized_data
def normalize(generated_x):
    x_min_np = np.array(x_min)
    x_max_np = np.array(x_max)

    generated_x = (generated_x - x_min_np) / (x_max_np - x_min_np)
    return generated_x

openai.api_key = "-"
file_nlp = "generated_descriptions.txt"
weights_file = "neural_network_weights.txt"
weights, biases = load_weights_from_text(weights_file)
W1, W2, W3 = weights
b1, b2, b3 = biases
x_min = train_X.min(axis=0)
x_max = train_X.max(axis=0)

attributes = list(mappings_dict.keys())
#print(mappings_dict)

def generate_input_for_class(desired_class, W1, b1, W2, b2, W3, b3, input_size, learning_rate=0.001, iterations=50000):
    x_prime = np.random.randn(1, input_size)
    x_prime = np.clip(x_prime, 0, 1)

    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, current_y = forward(x_prime, W1, b1, W2, b2, W3, b3, training=False)

        # loss
        target_output = np.zeros_like(current_y)
        target_output[0, desired_class] = 1.0
        loss = -np.log(current_y[0, desired_class] + 1e-10)

        # gradient fata de input
        d_output = current_y - target_output

        # backprop for x'
        dZ3 = d_output
        dA2 = np.dot(dZ3, W3.T)
        dZ2 = dA2 * relu_derivative(Z2)
        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * relu_derivative(Z1)
        x_prime_gradient = np.dot(dZ1, W1.T)

        # update x' with gradient
        x_prime -= learning_rate * x_prime_gradient
        x_prime = np.clip(x_prime, 0, 1)

        if i % 10000 == 0:
            print(f"Iteration {i}, Loss: {loss:.6f}")

    return x_prime

def generate_attributes_for_race(desired_class):
    breed_mapping = changes_df[changes_df["Column"] == "Race"].set_index("Original Value")["New Value"].to_dict()
    original_value = int(breed_mapping.get(desired_class , "Unknown")) - 1
    desired_class = original_value if original_value is not None else desired_class

    generated_x_normalized = generate_input_for_class(desired_class, W1, b1, W2, b2, W3, b3, input_size)
    _, _, _, _, _, generated_y = forward(generated_x_normalized, W1, b1, W2, b2, W3, b3, training=False)
    #print(f"Generated normalized input x': {generated_x_normalized}")
    #print(f"Generated output probabilities y': {generated_y}")
    #print(f"Predicted class: {np.argmax(generated_y)}")

    generated_x_normal_values = denormalize_and_clip(generated_x_normalized)

    #print("Generated input (denormalized):")
    #print(generated_x_normal_values[0])

    generated_normalized_x = normalize(generated_x_normal_values)
    _,_,_,_,_, test_pred_final = forward(generated_normalized_x, W1, b1, W2, b2, W3, b3, training=False)

    final_test_predictions = np.argmax(test_pred_final)
    print(f"Prediction: {final_test_predictions}")
    breed_mapping_reverse = changes_df[changes_df["Column"] == "Race"].set_index("New Value")["Original Value"].to_dict()
    predicted_breed = breed_mapping_reverse.get(final_test_predictions + 1, "Unknown")
    print(f"Prediction: {predicted_breed}")

    return generated_x_normal_values[0]


#description in natural language of cat
def generate_description(input_attributes):
    prompt = (
            "Generate a detailed description of a cat based on the following attributes:\n"
            + "\n".join(input_attributes)
            + "\n\nImportant constraints:\n"
              "- Use the exact values provided for each attribute.\n"
              "- Avoid synonyms or alternate phrasings for attribute values (e.g., 'a little' must not be replaced by 'somewhat').\n"
              "- Ensure the description strictly adheres to the input attributes.\n"
              "- Do not add interpretations or extra details not implied by the attributes.\n"
              "- Use every attribute.\n"
              "- Don't use words such as considerably for considerable and moderately for moderate."
              "- For each attribute that you use, you need to add the value next to him."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )

        content = response['choices'][0]['message']['content'].strip()
        #print(f"GPT response content: {content}")

        with open(file_nlp, 'a') as file:
            file.write(str(input_attributes) + "\n")

        with open(file_nlp, 'a') as file:
            file.write(content + "\n")

        return content
    except Exception as e:
        print(f"Error: {e}")
        return {}

def generate_desc(values_vector):
    description_data = []
    for i, value in enumerate(values_vector):
        attribute = attributes[i]
        for k, v in mappings_dict[attribute].items():
            if v == value:
                description_data.append(f"{attribute}: {k}")
                break
    return description_data

def choose_race():
    race = "Chartreux"
    values_vector = generate_attributes_for_race(race)
    print(values_vector)
    description_data = generate_desc(values_vector)
    print(description_data)

    # with open(file_nlp, 'a') as file:
    #   file.write("\n" + str(values_vector) + "\n")

    #print(generate_description(description_data))
#choose_race()


#for comparison
def generate_comparison(input_attributes1, race1, input_attributes2, race2):
    prompt = (
            f"Generate a natural language comparison between 2 cats based on the characteristics they have. The first cat has the race {race1} and has the following traits:\n"
            + "\n".join(input_attributes1)
            + f"The second cat has the race {race2} and has the following traits:\n"
            + "\n".join(input_attributes2)
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant trained to make comparison between 2 races of cats."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )

        content = response['choices'][0]['message']['content'].strip()
        print(f"GPT response content: {content}")

        return content
    except Exception as e:
        print(f"Error: {e}")
        return {}

def choose_2_breeds():
    race1 = "Chartreux"
    values_vector1 = generate_attributes_for_race(race1)
    print(values_vector1)
    race2 = "European"
    values_vector2 = generate_attributes_for_race(race2)
    print(values_vector2)

    features1 = generate_desc(values_vector1)
    features2 = generate_desc(values_vector2)

    print(features1)
    print(features2)
    generate_comparison(features1, race1, features2, race2)
#choose_2_breeds()





