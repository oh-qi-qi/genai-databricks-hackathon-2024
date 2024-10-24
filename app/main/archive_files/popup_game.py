import pygame
import random
from PIL import Image
import streamlit as st
from streamlit_modal import Modal
import threading
import time

# Initialize Pygame
pygame.init()

# Game constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 200
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Paddle constants
PADDLE_WIDTH = 60
PADDLE_HEIGHT = 10
PADDLE_SPEED = 5

# Ball constants
BALL_SIZE = 10
BALL_SPEED = 3

# Brick constants
BRICK_WIDTH = 38
BRICK_HEIGHT = 15
BRICK_ROWS = 3
BRICK_COLS = 10

class GameState:
    def __init__(self):
        self.paddle_x = SCREEN_WIDTH // 2 - PADDLE_WIDTH // 2
        self.ball_x = SCREEN_WIDTH // 2
        self.ball_y = SCREEN_HEIGHT - PADDLE_HEIGHT - BALL_SIZE - 1
        self.ball_dx = random.choice([-1, 1]) * BALL_SPEED
        self.ball_dy = -BALL_SPEED
        self.score = 0
        self.bricks = [[1 for _ in range(BRICK_COLS)] for _ in range(BRICK_ROWS)]
        self.game_over = False
        self.move_direction = None
        self.lock = threading.Lock()

def initialize_game_state():
    if 'game_state' not in st.session_state:
        st.session_state.game_state = GameState()
    if 'game_thread' not in st.session_state:
        st.session_state.game_thread = None

def reset_game():
    with st.session_state.game_state.lock:
        st.session_state.game_state = GameState()

def update_game_state():
    state = st.session_state.game_state
    with state.lock:
        # Move paddle
        if state.move_direction == 'left':
            state.paddle_x = max(0, state.paddle_x - PADDLE_SPEED)
        elif state.move_direction == 'right':
            state.paddle_x = min(SCREEN_WIDTH - PADDLE_WIDTH, state.paddle_x + PADDLE_SPEED)
        state.move_direction = None  # Reset move direction

        # Move ball
        state.ball_x += state.ball_dx
        state.ball_y += state.ball_dy

        # Ball collision with walls
        if state.ball_x <= 0 or state.ball_x >= SCREEN_WIDTH - BALL_SIZE:
            state.ball_dx *= -1
        if state.ball_y <= 0:
            state.ball_dy *= -1

        # Ball collision with paddle
        if (state.ball_y >= SCREEN_HEIGHT - PADDLE_HEIGHT - BALL_SIZE and
            state.paddle_x < state.ball_x < state.paddle_x + PADDLE_WIDTH):
            state.ball_dy *= -1
            state.ball_dx += random.uniform(-0.5, 0.5)  # Add some randomness

        # Ball collision with bricks
        for row in range(BRICK_ROWS):
            for col in range(BRICK_COLS):
                if state.bricks[row][col]:
                    brick_x = col * (BRICK_WIDTH + 2)
                    brick_y = row * (BRICK_HEIGHT + 2) + 30
                    if (brick_x < state.ball_x < brick_x + BRICK_WIDTH and
                        brick_y < state.ball_y < brick_y + BRICK_HEIGHT):
                        state.bricks[row][col] = 0
                        state.ball_dy *= -1
                        state.score += 1

        # Check for game over
        if state.ball_y >= SCREEN_HEIGHT:
            state.game_over = True

def render_game():
    state = st.session_state.game_state
    with state.lock:
        screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        screen.fill(BLACK)
        pygame.draw.rect(screen, WHITE, (state.paddle_x, SCREEN_HEIGHT - PADDLE_HEIGHT, PADDLE_WIDTH, PADDLE_HEIGHT))
        pygame.draw.circle(screen, WHITE, (int(state.ball_x), int(state.ball_y)), BALL_SIZE // 2)
        for row in range(BRICK_ROWS):
            for col in range(BRICK_COLS):
                if state.bricks[row][col]:
                    pygame.draw.rect(screen, RED, (col * (BRICK_WIDTH + 2), row * (BRICK_HEIGHT + 2) + 30, BRICK_WIDTH, BRICK_HEIGHT))

    return Image.frombytes('RGB', screen.get_size(), pygame.image.tostring(screen, 'RGB'))

def game_loop():
    clock = pygame.time.Clock()
    while not st.session_state.game_state.game_over:
        update_game_state()
        clock.tick(FPS)

def start_game_thread():
    if st.session_state.game_thread is None or not st.session_state.game_thread.is_alive():
        st.session_state.game_thread = threading.Thread(target=game_loop)
        st.session_state.game_thread.start()

def play_mini_breakout():
    initialize_game_state()
    state = st.session_state.game_state

    modal = Modal(title="Mini Breakout", key="game_modal")
    open_modal = st.button("Play Mini Breakout")

    if open_modal:
        modal.open()

    if modal.is_open():
        with modal.container():
            st.write("Use the buttons to move the paddle and break the bricks!")

            col1, col2, col3 = st.columns([1,3,1])
            with col1:
                if st.button("← Left"):
                    with state.lock:
                        state.move_direction = 'left'
            with col3:
                if st.button("Right →"):
                    with state.lock:
                        state.move_direction = 'right'

            start_game_thread()
            game_image = render_game()
            st.image(game_image, use_column_width=True)
            st.write(f"Score: {state.score}")

            if state.game_over:
                st.success(f"Game Over! Your final score: {state.score}")
                if st.button("Restart Game"):
                    reset_game()
                    st.experimental_rerun()

    return state.game_over, state.score

if __name__ == "__main__":
    play_mini_breakout()