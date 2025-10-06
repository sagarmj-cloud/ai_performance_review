"""
Authentication module for performance review system.
"""
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from pathlib import Path


class AuthManager:
    """Manages authentication for the performance review system."""
    
    def __init__(self):
        """Initialize authentication manager."""
        self.config = self._load_config()
        self.authenticator = self._create_authenticator()
    
    def _load_config(self):
        """
        Load authentication configuration.
        Tries Streamlit secrets first (cloud), then local YAML file.
        """
        try:
            # Try loading from Streamlit secrets (cloud deployment)
            if "credentials" in st.secrets:
                logger = __import__('logging').getLogger(__name__)
                logger.info("Loading auth config from Streamlit secrets")
                return dict(st.secrets)
        except (AttributeError, FileNotFoundError):
            pass
        
        # Fall back to local YAML file (local development)
        config_path = Path(".streamlit/config.yaml")
        if config_path.exists():
            with open(config_path) as file:
                return yaml.load(file, Loader=SafeLoader)
        
        raise FileNotFoundError(
            "Authentication config not found. Please create either:\n"
            "- .streamlit/secrets.toml (for Streamlit Cloud)\n"
            "- .streamlit/config.yaml (for local development)"
        )
    
    def _create_authenticator(self):
        """Create streamlit-authenticator instance."""
        return stauth.Authenticate(
            self.config['credentials'],
            self.config['cookie']['name'],
            self.config['cookie']['key'],
            self.config['cookie']['expiry_days']
        )
    
    def login(self):
        """Display login form and handle authentication."""
        return self.authenticator.login()
    
    def logout(self):
        """Display logout button."""
        self.authenticator.logout()
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated."""
        return st.session_state.get("authentication_status", False)
    
    def get_username(self) -> str:
        """Get authenticated username."""
        return st.session_state.get("username", "")
    
    def get_name(self) -> str:
        """Get authenticated user's full name."""
        return st.session_state.get("name", "")
    
    def get_roles(self) -> list:
        """Get user's roles."""
        username = self.get_username()
        if username:
            user_data = self.config['credentials']['usernames'].get(username, {})
            return user_data.get('roles', [])
        return []
    
    def get_employee_id(self) -> str:
        """Get user's employee ID from config."""
        username = self.get_username()
        if username:
            user_data = self.config['credentials']['usernames'].get(username, {})
            return user_data.get('employee_id', username)
        return ""
    
    def has_role(self, role: str) -> bool:
        """Check if user has specific role."""
        return role in self.get_roles()
    
    def require_authentication(self):
        """Require authentication or stop execution."""
        if not self.is_authenticated():
            st.warning("Please log in to access this page")
            st.stop()
    
    def require_role(self, required_role: str):
        """Require specific role or stop execution."""
        if not self.has_role(required_role):
            st.error(f"Access denied. {required_role.title()} role required.")
            st.stop()