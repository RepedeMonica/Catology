import tkinter as tk
from tkinter import Toplevel, messagebox
from PIL import Image, ImageTk
import random
from inverse import generate_attributes_for_race, generate_desc, generate_description

ALLOWED_BREEDS = [
    "Bengal", "Birman", "British Shorthair", "Chartreux", "European",
    "Maine Coon", "Persian", "Ragdol", "Sphynx", "Siamese", "Turkish Angora"
]

class CatQuizApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cat Quiz")
        self.root.geometry("1000x700")

        try:
            self.bg_image_main = Image.open("cats.jpg")
            self.bg_image_main = self.bg_image_main.resize((1000, 800), Image.Resampling.LANCZOS)
            self.bg_photo_main = ImageTk.PhotoImage(self.bg_image_main)

            self.bg_image_quiz = Image.open("cats2.jpg")
            self.bg_image_quiz = self.bg_image_quiz.resize((1000, 800), Image.Resampling.LANCZOS)
            self.bg_photo_quiz = ImageTk.PhotoImage(self.bg_image_quiz)

            self.bg_image_fact = Image.open("cats3.jpg")
            self.bg_image_fact = self.bg_image_fact.resize((600, 400), Image.Resampling.LANCZOS)
            self.bg_photo_fact = ImageTk.PhotoImage(self.bg_image_fact)
        except FileNotFoundError:
            self.bg_photo_main = None
            self.bg_photo_quiz = None
            self.bg_photo_fact = None


        self.canvas = tk.Canvas(self.root, width=1000, height=800)
        self.canvas.pack(fill="both", expand=True)

        if self.bg_photo_main:
            self.canvas.create_image(0, 0, image=self.bg_photo_main, anchor="nw")

        self.quiz_button = tk.Button(self.root, text="Quiz", command=self.open_quiz_screen, font=("Arial", 16, "bold"),
                                     bg="orange", width=12, height=1)
        self.fun_facts_button = tk.Button(self.root, text="Fun Facts", command=self.show_fun_facts,
                                          font=("Arial", 16, "bold"), bg="green", width=12, height=1)


        self.canvas.create_window(400, 400, window=self.quiz_button)
        self.canvas.create_window(600, 400, window=self.fun_facts_button)

    def open_quiz_screen(self):
        self.canvas.destroy()
        self.show_loading_screen()
        self.quiz_screen()

    def show_loading_screen(self):
        self.canvas = tk.Canvas(self.root, width=1000, height=800, bg="lightblue")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_text(500, 400, text="Loading...", font=("Arial", 24, "bold"), fill="darkblue")
        self.root.update()

    def quiz_screen(self):
        self.canvas.destroy()
        self.canvas = tk.Canvas(self.root, width=1000, height=800, bg="lightblue")
        self.canvas.pack(fill="both", expand=True)

        if self.bg_photo_quiz:
            self.canvas.create_image(0, 0, image=self.bg_photo_quiz, anchor="nw")

        self.canvas.create_rectangle(250, 50, 750, 130, fill="white", outline="darkblue", width=3)
        self.canvas.create_text(500, 90, text="GUESS THE CAT", font=("Arial", 28, "bold"), fill="darkblue")

        self.correct_breed = random.choice(ALLOWED_BREEDS)

        values_vector = generate_attributes_for_race(self.correct_breed)
        attributes = generate_desc(values_vector)
        self.description = generate_description(attributes)

        text_id = self.canvas.create_text(500, 250, text=self.description, font=("Arial", 16), fill="black", width=800,
                                          anchor="center")
        bbox = self.canvas.bbox(text_id)

        if bbox:
            self.canvas.create_rectangle(bbox, fill="white", outline="darkblue", width=3)
        self.canvas.lift(text_id)

        wrong_breeds = random.sample([breed for breed in ALLOWED_BREEDS if breed != self.correct_breed], 3)
        options = wrong_breeds + [self.correct_breed]
        random.shuffle(options)

        x_positions = [200, 400, 600, 800]
        y_position = 600

        for i, breed in enumerate(options):
            button = tk.Button(self.root, text=breed, font=("Arial", 14), bg="white", width=15,
                               command=lambda b=breed: self.check_answer(b))
            self.canvas.create_window(x_positions[i], y_position, window=button)

    def check_answer(self, chosen_breed):
        if chosen_breed == self.correct_breed:
            messagebox.showinfo("CORRECT!", "Good job!You guessed the correct breed!")
            self.open_main_menu()

        else:
            messagebox.showerror("WRONG!", "Try again!")

    def open_main_menu(self):
        self.canvas.destroy()
        self.__init__(self.root)

    def show_fun_facts(self):
        def filter_extremely_attributes(attributes, random_breed):
            extremely_attributes = [
                attribute.split(":")[0].strip()
                for attribute in attributes
                if "extremely" in attribute
            ]
            if not extremely_attributes:
                return f"The breed {random_breed} does not have any attributes described as extremely."

            attributes_sentence = ", ".join(extremely_attributes)
            return f"The breed {random_breed} is extremely {attributes_sentence}."

        random_breed = random.choice(ALLOWED_BREEDS)

        try:
            values_vector = generate_attributes_for_race(random_breed)
            attributes = generate_desc(values_vector)
        except ValueError as e:
            messagebox.showerror("Error", f"Failed to generate attributes: {e}")
            return

        extremely_attributes = filter_extremely_attributes(attributes, random_breed)

        fact_text = "".join(extremely_attributes) if extremely_attributes else "This breed has no extreme attributes."

        fact_window = Toplevel(self.root)
        fact_window.title("Fun Fact")
        fact_window.geometry("600x400")

        fact_canvas = tk.Canvas(fact_window, width=600, height=400)
        fact_canvas.pack(fill="both", expand=True)

        if self.bg_photo_fact:
            fact_canvas.create_image(0, 0, image=self.bg_photo_fact, anchor="nw")

        fact_canvas.create_rectangle(50, 20, 550, 120, fill="white", outline="darkblue", width=3)
        fact_canvas.create_text(300, 70, text=fact_text, font=("Arial", 14, "bold"), fill="black", width=480)

        exit_button = tk.Button(fact_window, text="Exit", font=("Arial", 12), bg="red", fg="white",
                                command=fact_window.destroy)
        fact_canvas.create_window(550, 350, window=exit_button)
if __name__ == "__main__":
    root = tk.Tk()
    app = CatQuizApp(root)
    root.mainloop()
