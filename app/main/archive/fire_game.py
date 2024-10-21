import random
import streamlit as st
import time

def fire_extinguishing_game():
    # The game state can be stored in session state
    if 'fires' not in st.session_state:
        st.session_state.fires = [random.randint(50, 350) for _ in range(10)]  # 10 random fire emojis
        st.session_state.extinguished_fires = []

    # Custom CSS to change the cursor to an extinguisher emoji
    st.markdown("""
        <style>
        .fire-emoji {
            font-size: 40px;
            position: absolute;
            cursor: url('https://twemoji.maxcdn.com/v/latest/72x72/1f9ef.png'), auto;
        }
        </style>
    """, unsafe_allow_html=True)

    # Create an area for the game
    with st.container():
        st.subheader("Click on the fires to extinguish them!")
        game_container = st.empty()
        
        with game_container:
            # Display fires
            fire_positions = st.session_state.fires
            extinguished = st.session_state.extinguished_fires

            for idx, fire in enumerate(fire_positions):
                if idx not in extinguished:
                    st.markdown(f'<div class="fire-emoji" style="left: {fire}px; top: {random.randint(50, 350)}px">🔥</div>', unsafe_allow_html=True)
            
            # Extinguish fires on click
            clicked_fire_idx = st.button("Extinguish")
            if clicked_fire_idx is not None and clicked_fire_idx not in extinguished:
                st.session_state.extinguished_fires.append(clicked_fire_idx)
            
            # End game logic
            if len(st.session_state.extinguished_fires) == len(st.session_state.fires):
                st.success("All fires extinguished!")
                return True  # Game finished
    return False  # Game still ongoing
