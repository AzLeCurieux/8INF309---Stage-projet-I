"""
Shared Authlib OAuth instance — initialized later via oauth.init_app(app).
Import this object in both server.py and auth.py to avoid circular imports.
"""
from authlib.integrations.flask_client import OAuth

oauth = OAuth()
