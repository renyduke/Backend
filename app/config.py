import os
from dotenv import load_dotenv

load_dotenv()

# Resolve the repo root reliably regardless of working directory
# app/config.py -> app/ -> repo root
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_APP_DIR)

class Settings:
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    PORT: int = int(os.getenv("PORT", 8000))
    BASE_DIR: str = _REPO_ROOT
    MODELS_DIR: str = os.getenv("MODELS_DIR", os.path.join(_REPO_ROOT, "models"))
    PLOT_DIR: str = os.getenv("PLOT_DIR", os.path.join(_REPO_ROOT, "plots"))
    
    def validate(self):
        if not self.SUPABASE_URL or not self.SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env file")

settings = Settings()