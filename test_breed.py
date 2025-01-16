import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from identify_gpt import mappings_dict, load_weights_from_text, changes_df
from nn import forward, train_X


class CatBreedPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cat Breed Predictor")
        self.root.geometry("800x600")

        tk.Label(self.root, text="Cat Breed Predictor", font=("Arial", 20, "bold")).pack(pady=10)

        scroll_frame = tk.Frame(self.root)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(scroll_frame)
        self.canvas.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(scroll_frame, orient="vertical", command=self.canvas.yview)
        scrollbar.pack(side="right", fill="y")

        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.scrollable_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.attribute_values = {}

        for i, (attribute, options) in enumerate(mappings_dict.items()):
            tk.Label(self.scrollable_frame, text=attribute, font=("Arial", 12), anchor="w").grid(row=i, column=0, padx=10, pady=5, sticky="w")
            combobox = ttk.Combobox(self.scrollable_frame, values=list(options.keys()), font=("Arial", 10))
            combobox.set("Select")
            combobox.grid(row=i, column=1, padx=10, pady=5)
            self.attribute_values[attribute] = combobox

        predict_button = tk.Button(self.root, text="Predict Breed", font=("Arial", 14), bg="green", fg="white",
                                   command=self.predict_breed_action)
        predict_button.pack(pady=20)

        self.result_label = tk.Label(self.root, text="", font=("Arial", 14), fg="blue")
        self.result_label.pack(pady=10)

    def predict_breed_action(self):
        input_vector = []
        for attribute, combobox in self.attribute_values.items():
            selected_value = combobox.get()
            if selected_value == "Select":
                messagebox.showerror("Error", f"Please select a value for {attribute}.")
                return
            input_vector.append(mappings_dict[attribute][selected_value])

        print(input_vector)
        x_min = train_X.min(axis=0).to_numpy()
        x_max = train_X.max(axis=0).to_numpy()

        normalized_input = (input_vector - x_min) / (x_max - x_min)
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

        self.result_label.config(text=f"The predicted breed is: {predicted_breed}")


if __name__ == "__main__":
    root = tk.Tk()
    app = CatBreedPredictorApp(root)
    root.mainloop()
