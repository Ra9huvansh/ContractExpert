import streamlit as st
import pandas as pd
import hashlib

# Hashing function to store passwords securely
def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()

# Validate login credentials
def validate_login(username, password, user_data):
    hashed_password = hash_password(password)
    user = user_data[(user_data['username'] == username) & (user_data['password'] == hashed_password)]
    return not user.empty

# Register a new user
def register_user(username, password, user_data_path):
    try:
        user_data = pd.read_csv(user_data_path)
    except FileNotFoundError:
        user_data = pd.DataFrame(columns=["username", "password"])

    if username in user_data['username'].values:
        return False, "Username already exists!"
    
    hashed_password = hash_password(password)
    new_user = pd.DataFrame({"username": [username], "password": [hashed_password]})
    new_user.to_csv(user_data_path, mode='a', header=False, index=False)
    return True, "Registration successful!"

# Login page
def login_page(user_data_path="users.csv"):
    st.title("Login Page")
    tab1, tab2 = st.tabs(["Login", "Register"])

    # Login Tab
    with tab1:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            try:
                user_data = pd.read_csv(user_data_path)
                if validate_login(username, password, user_data):
                    st.success(f"Welcome {username}!")
                    return username  # Return the logged-in username
                else:
                    st.error("Invalid username or password.")
            except FileNotFoundError:
                st.error("User database not found!")

    # Registration Tab
    with tab2:
        st.subheader("Register")
        reg_username = st.text_input("New Username")
        reg_password = st.text_input("New Password", type="password")
        if st.button("Register"):
            success, message = register_user(reg_username, reg_password, user_data_path)
            if success:
                st.success(message)
            else:
                st.error(message)

    return None



# Logout button functionality

def logout_button():
    """Displays a logout button and resets the login state when pressed."""
    if st.button("Logout"):
        # Reset session state for login
        st.session_state['logged_in'] = False
        st.experimental_rerun()  # Redirect back to the login page

