"""
Authentication Blueprint — login, signup, logout, email verification, password reset.
"""
import os
import logging

from flask import (Blueprint, flash, redirect, render_template,
                   request, url_for, current_app)
from flask_login import login_user, logout_user, login_required, current_user
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature

from models import (create_user, get_user_by_email, get_user_by_id,
                    get_user_by_oauth, create_oauth_user,
                    update_last_login, verify_user, update_password,
                    update_profile, update_newsletter, log_activity)
from oauth_client import oauth

auth = Blueprint("auth", __name__, url_prefix="/auth")

TOKEN_MAX_AGE_VERIFY = 86400   # 24 h
TOKEN_MAX_AGE_RESET  = 3600    # 1 h


def _serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(current_app.secret_key)


def _send_email(to: str, subject: str, html_body: str) -> bool:
    """Send an email via Flask-Mail. Returns False (logs warning) if mail not configured."""
    try:
        mail = current_app.extensions.get("mail")
        if mail is None:
            logging.warning("[auth] Flask-Mail not configured — email not sent to %s", to)
            return False
        from flask_mail import Message
        msg = Message(subject=subject,
                      recipients=[to],
                      html=html_body,
                      sender=current_app.config.get("MAIL_DEFAULT_SENDER", "noreply@chickenwings.local"))
        mail.send(msg)
        return True
    except Exception as exc:
        logging.warning("[auth] Failed to send email to %s: %s", to, exc)
        return False


def _send_verification(user):
    token = _serializer().dumps(user.email, salt="email-verify")
    link = url_for("auth.verify_email", token=token, _external=True)
    html = render_template("email/verify.html", user=user, link=link)
    _send_email(user.email, "Confirm your Chicken Wings account", html)
    return link  # returned for dev/logging


def _send_password_reset(user):
    token = _serializer().dumps(user.email, salt="pw-reset")
    link = url_for("auth.reset_password", token=token, _external=True)
    html = render_template("email/reset_password.html", user=user, link=link)
    _send_email(user.email, "Reset your Chicken Wings password", html)
    return link


# ---------------------------------------------------------------------------
# Login / Logout
# ---------------------------------------------------------------------------

@auth.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        remember = request.form.get("remember") == "on"

        user = get_user_by_email(email)
        if not user or not user.check_password(password):
            error = "Invalid email or password."
            log_activity(None, email, "login_failed", f"ip={request.remote_addr}", request.remote_addr)
        else:
            login_user(user, remember=remember)
            update_last_login(user.id)
            log_activity(user.id, user.email, "login", "", request.remote_addr)
            next_page = request.args.get("next") or url_for("dashboard")
            return redirect(next_page)

    return render_template("login.html", error=error)


@auth.route("/logout")
@login_required
def logout():
    log_activity(current_user.id, current_user.email, "logout", "", request.remote_addr)
    logout_user()
    return redirect(url_for("auth.login"))


# ---------------------------------------------------------------------------
# Sign up
# ---------------------------------------------------------------------------

@auth.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    error = None
    if request.method == "POST":
        email      = request.form.get("email", "").strip().lower()
        password   = request.form.get("password", "")
        password2  = request.form.get("password2", "")
        first_name = request.form.get("first_name", "").strip()
        last_name  = request.form.get("last_name", "").strip()

        if not email or "@" not in email:
            error = "Please enter a valid email address."
        elif len(password) < 8:
            error = "Password must be at least 8 characters."
        elif password != password2:
            error = "Passwords do not match."
        else:
            user = create_user(email, password, first_name, last_name)
            if user is None:
                error = "An account with this email already exists."
            else:
                link = _send_verification(user)
                logging.info("[auth] New user %s — verification link: %s", email, link)
                log_activity(user.id, user.email, "signup", "", request.remote_addr)
                return render_template("auth/signup_success.html", email=email)

    return render_template("signup.html", error=error)


# ---------------------------------------------------------------------------
# Email verification
# ---------------------------------------------------------------------------

@auth.route("/verify-email/<token>")
def verify_email(token: str):
    try:
        email = _serializer().loads(token, salt="email-verify", max_age=TOKEN_MAX_AGE_VERIFY)
    except SignatureExpired:
        return render_template("auth/verify_result.html",
                               success=False, reason="expired")
    except BadSignature:
        return render_template("auth/verify_result.html",
                               success=False, reason="invalid")

    user = get_user_by_email(email)
    if not user:
        return render_template("auth/verify_result.html",
                               success=False, reason="not_found")

    if user.is_verified:
        return render_template("auth/verify_result.html",
                               success=True, already=True)

    verify_user(user.id)
    log_activity(user.id, user.email, "email_verified", "", request.remote_addr)
    return render_template("auth/verify_result.html", success=True, already=False)


@auth.route("/resend-verification", methods=["POST"])
@login_required
def resend_verification():
    if current_user.is_verified:
        return redirect(url_for("dashboard"))
    link = _send_verification(current_user)
    logging.info("[auth] Resent verification to %s — %s", current_user.email, link)
    flash("Verification email sent!", "success")
    return redirect(url_for("dashboard"))


# ---------------------------------------------------------------------------
# Forgot / reset password
# ---------------------------------------------------------------------------

@auth.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    sent = False
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        user = get_user_by_email(email)
        if user:
            link = _send_password_reset(user)
            logging.info("[auth] Password reset link for %s: %s", email, link)
            log_activity(user.id, user.email, "password_reset_requested", "", request.remote_addr)
        # Always show success to avoid user enumeration
        sent = True

    return render_template("forgot_password.html", sent=sent)


@auth.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token: str):
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    error = None
    try:
        email = _serializer().loads(token, salt="pw-reset", max_age=TOKEN_MAX_AGE_RESET)
    except SignatureExpired:
        return render_template("reset_password.html",
                               token=token, error="This link has expired. Please request a new one.", expired=True)
    except BadSignature:
        return render_template("reset_password.html",
                               token=token, error="Invalid reset link.", expired=True)

    user = get_user_by_email(email)
    if not user:
        return render_template("reset_password.html",
                               token=token, error="Account not found.", expired=True)

    if request.method == "POST":
        password  = request.form.get("password", "")
        password2 = request.form.get("password2", "")
        if len(password) < 8:
            error = "Password must be at least 8 characters."
        elif password != password2:
            error = "Passwords do not match."
        else:
            update_password(user.id, password)
            log_activity(user.id, user.email, "password_reset", "", request.remote_addr)
            return render_template("reset_password.html", token=token, done=True)

    return render_template("reset_password.html", token=token, error=error, expired=False)


# ---------------------------------------------------------------------------
# OAuth SSO — Google & GitHub
# ---------------------------------------------------------------------------

def _sso_login_or_create(email: str, first_name: str, last_name: str,
                          provider: str, oauth_id: str):
    """Find or create a user from an OAuth callback, then log them in."""
    # 1. Try matching by provider + oauth_id
    user = get_user_by_oauth(provider, oauth_id)
    # 2. Fall back to email match (link existing account)
    if not user:
        user = get_user_by_email(email)
    # 3. Create new account
    if not user:
        user = create_oauth_user(email, first_name, last_name, provider, oauth_id)
    if not user:
        return redirect(url_for("auth.login") + "?error=oauth_failed")
    login_user(user)
    update_last_login(user.id)
    log_activity(user.id, user.email, f"login_sso_{provider}", "", "")
    return redirect(url_for("dashboard"))


@auth.route("/google")
def google_login():
    redirect_uri = url_for("auth.google_callback", _external=True)
    return oauth.google.authorize_redirect(redirect_uri)


@auth.route("/google/callback")
def google_callback():
    try:
        token = oauth.google.authorize_access_token()
        info = token.get("userinfo") or oauth.google.userinfo()
        return _sso_login_or_create(
            email      = info.get("email", ""),
            first_name = info.get("given_name", ""),
            last_name  = info.get("family_name", ""),
            provider   = "google",
            oauth_id   = str(info.get("sub", "")),
        )
    except Exception as exc:
        logging.warning("[auth] Google OAuth error: %s", exc)
        return redirect(url_for("auth.login") + "?error=oauth_failed")


@auth.route("/github")
def github_login():
    redirect_uri = url_for("auth.github_callback", _external=True)
    return oauth.github.authorize_redirect(redirect_uri)


@auth.route("/github/callback")
def github_callback():
    try:
        oauth.github.authorize_access_token()
        resp = oauth.github.get("user")
        info = resp.json()
        email = info.get("email") or ""
        # GitHub may hide the email — fetch from /user/emails
        if not email:
            emails_resp = oauth.github.get("user/emails")
            for e in emails_resp.json():
                if e.get("primary") and e.get("verified"):
                    email = e["email"]
                    break
        name_parts = (info.get("name") or "").split(" ", 1)
        return _sso_login_or_create(
            email      = email,
            first_name = name_parts[0] if name_parts else "",
            last_name  = name_parts[1] if len(name_parts) > 1 else "",
            provider   = "github",
            oauth_id   = str(info.get("id", "")),
        )
    except Exception as exc:
        logging.warning("[auth] GitHub OAuth error: %s", exc)
        return redirect(url_for("auth.login") + "?error=oauth_failed")


# ---------------------------------------------------------------------------
# Account settings (AJAX endpoints)
# ---------------------------------------------------------------------------

@auth.route("/account/profile", methods=["POST"])
@login_required
def update_profile_route():
    from flask import jsonify
    first_name = request.form.get("first_name", "").strip()
    last_name  = request.form.get("last_name", "").strip()
    ok = update_profile(current_user.id, first_name, last_name)
    if ok:
        log_activity(current_user.id, current_user.email, "profile_updated", "", request.remote_addr)
    return jsonify({"ok": ok})


@auth.route("/account/password", methods=["POST"])
@login_required
def change_password():
    from flask import jsonify
    current_pw = request.form.get("current_password", "")
    new_pw     = request.form.get("new_password", "")
    new_pw2    = request.form.get("new_password2", "")

    if not current_user.check_password(current_pw):
        return jsonify({"ok": False, "error": "Current password is incorrect."})
    if len(new_pw) < 8:
        return jsonify({"ok": False, "error": "New password must be at least 8 characters."})
    if new_pw != new_pw2:
        return jsonify({"ok": False, "error": "Passwords do not match."})

    ok = update_password(current_user.id, new_pw)
    if ok:
        log_activity(current_user.id, current_user.email, "password_changed", "", request.remote_addr)
    return jsonify({"ok": ok})


@auth.route("/account/newsletter", methods=["POST"])
@login_required
def toggle_newsletter():
    from flask import jsonify
    subscribed = request.form.get("subscribed") == "1"
    ok = update_newsletter(current_user.id, subscribed)
    if ok:
        log_activity(current_user.id, current_user.email,
                     "newsletter_subscribed" if subscribed else "newsletter_unsubscribed", "", request.remote_addr)
    return jsonify({"ok": ok, "subscribed": subscribed})
