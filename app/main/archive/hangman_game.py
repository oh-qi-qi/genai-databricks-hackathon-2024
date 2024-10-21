# hangman_game.py
import random

words = [
    {"word": "TOWER", "hint": "Tall, fixed crane often used in high-rise construction"},
    {"word": "MOBILE", "hint": "Type of crane mounted on wheeled or tracked carriers"},
    {"word": "GANTRY", "hint": "Crane built atop a frame straddling an area"},
    {"word": "JIB", "hint": "Horizontal or angled extension of a crane's main boom"},
    {"word": "CRAWLER", "hint": "Crane mounted on an undercarriage with track-type wheels"},
    # ... add more words ...
]

def initialize_game():
    word_obj = random.choice(words)
    return {
        "word": word_obj["word"],
        "hint": word_obj["hint"],
        "guessed_letters": set(),
        "remaining_guesses": 6,
        "score": 0
    }

def guess_letter(game_state, letter):
    if letter not in game_state["guessed_letters"]:
        game_state["guessed_letters"].add(letter)
        if letter not in game_state["word"]:
            game_state["remaining_guesses"] -= 1
        else:
            game_state["score"] += 1
    return game_state

def get_masked_word(word, guessed_letters):
    return ''.join([letter if letter in guessed_letters else '_' for letter in word])

def is_game_over(game_state):
    return (game_state["remaining_guesses"] == 0 or 
            set(game_state["word"]) <= game_state["guessed_letters"])

def get_game_summary(game_state):
    return f"Game ended. Word: {game_state['word']}. Score: {game_state['score']}"