"""
User model and database helpers for authentication & user management.
"""
import os
import secrets
from datetime import datetime

import mysql.connector
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

_DB_CFG = dict(
    host=os.environ.get("DB_HOST", "127.0.0.1"),
    port=int(os.environ.get("DB_PORT", "3306")),
    user=os.environ.get("DB_USER", "root"),
    password=os.environ.get("DB_PASSWORD", "1234"),
    database=os.environ.get("DB_NAME", "promotions_db"),
)


def _get_db():
    return mysql.connector.connect(**_DB_CFG)


# ---------------------------------------------------------------------------
# User model
# ---------------------------------------------------------------------------

class User(UserMixin):
    def __init__(self, row: dict):
        self.id = row["id"]
        self.email = row["email"]
        self.password_hash = row["password_hash"]
        self.first_name = row.get("first_name") or ""
        self.last_name = row.get("last_name") or ""
        self.role = row.get("role", "user")
        self.is_verified = bool(row.get("is_verified", 0))
        self.newsletter_subscribed = bool(row.get("newsletter_subscribed", 1))
        self.unsubscribe_token = row.get("unsubscribe_token")
        self.created_at = row.get("created_at")
        self.last_login = row.get("last_login")

    def check_password(self, password: str) -> bool:
        if not self.password_hash:
            return False
        return check_password_hash(self.password_hash, password)

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def display_name(self) -> str:
        if self.first_name:
            return self.first_name
        return self.email.split("@")[0]

    @property
    def full_name(self) -> str:
        parts = [self.first_name, self.last_name]
        name = " ".join(p for p in parts if p)
        return name or self.email


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def get_user_by_id(user_id: int) -> User | None:
    try:
        db = _get_db()
        cur = db.cursor(dictionary=True)
        cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        row = cur.fetchone()
        cur.close(); db.close()
        return User(row) if row else None
    except Exception:
        return None


def get_user_by_email(email: str) -> User | None:
    try:
        db = _get_db()
        cur = db.cursor(dictionary=True)
        cur.execute("SELECT * FROM users WHERE email = %s", (email.lower().strip(),))
        row = cur.fetchone()
        cur.close(); db.close()
        return User(row) if row else None
    except Exception:
        return None


def create_user(email: str, password: str, first_name: str = "", last_name: str = "") -> User | None:
    """Insert new user. Returns User on success, None if email already taken."""
    try:
        db = _get_db()
        cur = db.cursor(dictionary=True)
        pw_hash = generate_password_hash(password)
        unsub_token = secrets.token_urlsafe(32)
        cur.execute(
            """INSERT INTO users
               (email, password_hash, first_name, last_name, role,
                is_verified, newsletter_subscribed, unsubscribe_token, created_at)
               VALUES (%s, %s, %s, %s, 'user', 0, 1, %s, %s)""",
            (email.lower().strip(), pw_hash, first_name.strip(), last_name.strip(),
             unsub_token, datetime.now()),
        )
        db.commit()
        new_id = cur.lastrowid
        cur.close(); db.close()
        return get_user_by_id(new_id)
    except mysql.connector.IntegrityError:
        return None  # duplicate email


def get_user_by_oauth(provider: str, oauth_id: str) -> "User | None":
    try:
        db = _get_db()
        cur = db.cursor(dictionary=True)
        cur.execute("SELECT * FROM users WHERE oauth_provider = %s AND oauth_id = %s",
                    (provider, oauth_id))
        row = cur.fetchone()
        cur.close(); db.close()
        return User(row) if row else None
    except Exception:
        return None


def create_oauth_user(email: str, first_name: str, last_name: str,
                      provider: str, oauth_id: str) -> "User | None":
    """Create a user authenticated via OAuth (no password)."""
    try:
        db = _get_db()
        cur = db.cursor(dictionary=True)
        unsub_token = secrets.token_urlsafe(32)
        cur.execute(
            """INSERT INTO users
               (email, password_hash, first_name, last_name, role,
                is_verified, newsletter_subscribed, unsubscribe_token,
                created_at, oauth_provider, oauth_id)
               VALUES (%s, NULL, %s, %s, 'user', 1, 1, %s, %s, %s, %s)""",
            (email.lower().strip(), first_name.strip(), last_name.strip(),
             unsub_token, datetime.now(), provider, oauth_id),
        )
        db.commit()
        new_id = cur.lastrowid
        cur.close(); db.close()
        return get_user_by_id(new_id)
    except mysql.connector.IntegrityError:
        return get_user_by_email(email)


def update_last_login(user_id: int):
    try:
        db = _get_db()
        cur = db.cursor()
        cur.execute("UPDATE users SET last_login = %s WHERE id = %s", (datetime.now(), user_id))
        db.commit(); cur.close(); db.close()
    except Exception:
        pass


def verify_user(user_id: int):
    try:
        db = _get_db()
        cur = db.cursor()
        cur.execute(
            "UPDATE users SET is_verified = 1, email_verified_at = %s WHERE id = %s",
            (datetime.now(), user_id),
        )
        db.commit(); cur.close(); db.close()
    except Exception:
        pass


def update_password(user_id: int, new_password: str):
    try:
        db = _get_db()
        cur = db.cursor()
        pw_hash = generate_password_hash(new_password)
        cur.execute("UPDATE users SET password_hash = %s WHERE id = %s", (pw_hash, user_id))
        db.commit(); cur.close(); db.close()
        return True
    except Exception:
        return False


def update_profile(user_id: int, first_name: str, last_name: str):
    try:
        db = _get_db()
        cur = db.cursor()
        cur.execute(
            "UPDATE users SET first_name = %s, last_name = %s WHERE id = %s",
            (first_name.strip(), last_name.strip(), user_id),
        )
        db.commit(); cur.close(); db.close()
        return True
    except Exception:
        return False


def update_newsletter(user_id: int, subscribed: bool):
    try:
        db = _get_db()
        cur = db.cursor()
        cur.execute("UPDATE users SET newsletter_subscribed = %s WHERE id = %s",
                    (1 if subscribed else 0, user_id))
        db.commit(); cur.close(); db.close()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Admin helpers
# ---------------------------------------------------------------------------

def get_all_users() -> list[dict]:
    try:
        db = _get_db()
        cur = db.cursor(dictionary=True)
        cur.execute("SELECT id, email, first_name, last_name, role, is_verified, "
                    "newsletter_subscribed, created_at, last_login FROM users ORDER BY created_at DESC")
        rows = cur.fetchall()
        cur.close(); db.close()
        return rows
    except Exception:
        return []


def set_user_role(user_id: int, role: str):
    if role not in ("user", "admin"):
        return False
    try:
        db = _get_db()
        cur = db.cursor()
        cur.execute("UPDATE users SET role = %s WHERE id = %s", (role, user_id))
        db.commit(); cur.close(); db.close()
        return True
    except Exception:
        return False


def delete_user(user_id: int):
    try:
        db = _get_db()
        cur = db.cursor()
        cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
        db.commit(); cur.close(); db.close()
        return True
    except Exception:
        return False


def get_subscribers() -> list[dict]:
    try:
        db = _get_db()
        cur = db.cursor(dictionary=True)
        cur.execute("SELECT id, email, first_name, last_name, created_at FROM users "
                    "WHERE newsletter_subscribed = 1 AND is_verified = 1 ORDER BY created_at DESC")
        rows = cur.fetchall()
        cur.close(); db.close()
        return rows
    except Exception:
        return []


def get_user_count() -> dict:
    try:
        db = _get_db()
        cur = db.cursor(dictionary=True)
        cur.execute("""SELECT
            COUNT(*) AS total,
            SUM(is_verified) AS verified,
            SUM(newsletter_subscribed) AS subscribers,
            SUM(role = 'admin') AS admins
            FROM users""")
        row = cur.fetchone()
        cur.close(); db.close()
        return row or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Activity logging
# ---------------------------------------------------------------------------

def log_activity(user_id: int | None, user_email: str, action: str,
                 details: str = "", ip: str = ""):
    try:
        db = _get_db()
        cur = db.cursor()
        cur.execute(
            """INSERT INTO activity_logs (user_id, user_email, action, details, ip_address, created_at)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (user_id, user_email, action, details, ip, datetime.now()),
        )
        db.commit(); cur.close(); db.close()
    except Exception:
        pass


def get_activity_logs(limit: int = 200, user_email: str | None = None) -> list[dict]:
    try:
        db = _get_db()
        cur = db.cursor(dictionary=True)
        if user_email:
            cur.execute(
                "SELECT * FROM activity_logs WHERE user_email = %s ORDER BY created_at DESC LIMIT %s",
                (user_email, limit),
            )
        else:
            cur.execute("SELECT * FROM activity_logs ORDER BY created_at DESC LIMIT %s", (limit,))
        rows = cur.fetchall()
        cur.close(); db.close()
        return rows
    except Exception:
        return []


# ---------------------------------------------------------------------------
# App settings (key-value store)
# ---------------------------------------------------------------------------

def get_setting(key: str, default: str = "") -> str:
    try:
        db = _get_db()
        cur = db.cursor(dictionary=True)
        cur.execute("SELECT value FROM app_settings WHERE `key` = %s", (key,))
        row = cur.fetchone()
        cur.close(); db.close()
        return row["value"] if row else default
    except Exception:
        return default


def set_setting(key: str, value: str) -> bool:
    try:
        db = _get_db()
        cur = db.cursor()
        cur.execute(
            "INSERT INTO app_settings (`key`, value, updated_at) VALUES (%s, %s, %s) "
            "ON DUPLICATE KEY UPDATE value = %s, updated_at = %s",
            (key, value, datetime.now(), value, datetime.now()),
        )
        db.commit(); cur.close(); db.close()
        return True
    except Exception:
        return False


def get_all_settings() -> dict:
    try:
        db = _get_db()
        cur = db.cursor(dictionary=True)
        cur.execute("SELECT `key`, value FROM app_settings")
        rows = cur.fetchall()
        cur.close(); db.close()
        return {r["key"]: r["value"] for r in rows}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Restaurant subscriptions
# ---------------------------------------------------------------------------

def subscribe_restaurant(user_id: int, restaurant_id: int, frequency: str = "weekly") -> bool:
    if frequency not in ("instant", "weekly", "monthly"):
        frequency = "weekly"
    try:
        db = _get_db()
        cur = db.cursor()
        cur.execute(
            "INSERT INTO restaurant_subscriptions (user_id, restaurant_id, frequency, created_at) "
            "VALUES (%s, %s, %s, %s) ON DUPLICATE KEY UPDATE frequency = %s",
            (user_id, restaurant_id, frequency, datetime.now(), frequency),
        )
        db.commit(); cur.close(); db.close()
        return True
    except Exception:
        return False


def unsubscribe_restaurant(user_id: int, restaurant_id: int) -> bool:
    try:
        db = _get_db()
        cur = db.cursor()
        cur.execute(
            "DELETE FROM restaurant_subscriptions WHERE user_id = %s AND restaurant_id = %s",
            (user_id, restaurant_id),
        )
        db.commit(); cur.close(); db.close()
        return True
    except Exception:
        return False


def get_user_subscriptions(user_id: int) -> list[dict]:
    try:
        db = _get_db()
        cur = db.cursor(dictionary=True)
        cur.execute(
            "SELECT rs.restaurant_id, rs.frequency, r.name, r.url "
            "FROM restaurant_subscriptions rs "
            "JOIN restaurants r ON r.id = rs.restaurant_id "
            "WHERE rs.user_id = %s",
            (user_id,),
        )
        rows = cur.fetchall()
        cur.close(); db.close()
        return rows
    except Exception:
        return []


def get_restaurant_subscribers(restaurant_id: int) -> list[dict]:
    """Return all verified users subscribed to a given restaurant."""
    try:
        db = _get_db()
        cur = db.cursor(dictionary=True)
        cur.execute(
            "SELECT u.id, u.email, u.first_name, u.last_name, u.unsubscribe_token, rs.frequency "
            "FROM restaurant_subscriptions rs "
            "JOIN users u ON u.id = rs.user_id "
            "WHERE rs.restaurant_id = %s AND u.is_verified = 1",
            (restaurant_id,),
        )
        rows = cur.fetchall()
        cur.close(); db.close()
        return rows
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Notification queue
# ---------------------------------------------------------------------------

def queue_notification(restaurant_id: int, restaurant_name: str, subject: str,
                       html_content: str, promo_count: int, status: str = "pending") -> int | None:
    try:
        db = _get_db()
        cur = db.cursor()
        cur.execute(
            "INSERT INTO notification_queue "
            "(restaurant_id, restaurant_name, subject, html_content, promo_count, status, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (restaurant_id, restaurant_name, subject, html_content, promo_count, status, datetime.now()),
        )
        db.commit()
        nid = cur.lastrowid
        cur.close(); db.close()
        return nid
    except Exception:
        return None


def get_notifications(limit: int = 100) -> list[dict]:
    try:
        db = _get_db()
        cur = db.cursor(dictionary=True)
        cur.execute(
            "SELECT id, restaurant_id, restaurant_name, subject, promo_count, status, created_at, sent_at "
            "FROM notification_queue ORDER BY created_at DESC LIMIT %s",
            (limit,),
        )
        rows = cur.fetchall()
        cur.close(); db.close()
        for r in rows:
            for k, v in r.items():
                if hasattr(v, "isoformat"):
                    r[k] = v.isoformat()
        return rows
    except Exception:
        return []


def get_approved_notifications() -> list[dict]:
    """Return all approved notifications not yet sent."""
    try:
        db = _get_db()
        cur = db.cursor(dictionary=True)
        cur.execute(
            "SELECT * FROM notification_queue WHERE status = 'approved' ORDER BY created_at ASC"
        )
        rows = cur.fetchall()
        cur.close(); db.close()
        return rows
    except Exception:
        return []


def update_notification_status(notif_id: int, status: str) -> bool:
    try:
        db = _get_db()
        cur = db.cursor()
        if status == "sent":
            cur.execute(
                "UPDATE notification_queue SET status = %s, sent_at = %s WHERE id = %s",
                (status, datetime.now(), notif_id),
            )
        else:
            cur.execute(
                "UPDATE notification_queue SET status = %s WHERE id = %s",
                (status, notif_id),
            )
        db.commit(); cur.close(); db.close()
        return True
    except Exception:
        return False


def get_notification_by_id(notif_id: int) -> dict | None:
    try:
        db = _get_db()
        cur = db.cursor(dictionary=True)
        cur.execute("SELECT * FROM notification_queue WHERE id = %s", (notif_id,))
        row = cur.fetchone()
        cur.close(); db.close()
        return row
    except Exception:
        return None
