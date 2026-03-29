"""
Shared decorators for route protection.
"""
from functools import wraps
from flask import redirect, url_for, abort, request
from flask_login import current_user


def admin_required(f):
    """Redirect to login if not authenticated; abort 403 if not admin."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for("auth.login", next=request.url))
        if not current_user.is_admin:
            abort(403)
        return f(*args, **kwargs)
    return decorated
