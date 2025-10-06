import streamlit_authenticator as stauth

# Correct structure that hash_passwords() expects
credentials = {
    'usernames':{
        'admin': {
            'password': 'admin123'
        },
        'jsmith': {
            'password': 'manager456'
        },
        'msarah': {
            'password': 'employee789'
        },
        'jtarsitano': {
            'password': 'employee790'
        }
    }
}

# Hash the passwords
hashed = stauth.Hasher.hash_passwords(credentials)

# Print the hashed passwords
for username, user_data in credentials['usernames'].items():
    print(f"{username}: {user_data['password']}")