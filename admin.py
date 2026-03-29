"""
Admin Blueprint — renders the admin panel UI.
All data is fetched from /api/admin/* endpoints defined in server.py.
"""
from flask import Blueprint, render_template
from decorators import admin_required

admin = Blueprint("admin", __name__, url_prefix="/admin")


@admin.route("/")
@admin_required
def panel():
    return render_template("admin.html")
