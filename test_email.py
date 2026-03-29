"""
Quick test: send a promo digest email via the configured SMTP (Resend).
Usage: docker compose exec app python test_email.py
"""
import os, smtplib, mysql.connector
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()

MAIL_SERVER = os.environ.get("MAIL_SERVER", "")
MAIL_PORT   = int(os.environ.get("MAIL_PORT", "465"))
MAIL_USER   = os.environ.get("MAIL_USERNAME", "")
MAIL_PASS   = os.environ.get("MAIL_PASSWORD", "")
MAIL_SENDER = os.environ.get("MAIL_DEFAULT_SENDER", "noreply@azwired.online")
TO          = "azsonmail@gmail.com"

# Fetch active promos from DB
db = mysql.connector.connect(
    host=os.environ.get("DB_HOST", "db"),
    port=int(os.environ.get("DB_PORT", "3306")),
    user=os.environ.get("DB_USER", "root"),
    password=os.environ.get("DB_PASSWORD", "1234"),
    database=os.environ.get("DB_NAME", "promotions_db"),
)
cur = db.cursor(dictionary=True)
cur.execute("""
    SELECT restaurant, promo_type, promo_details, link, image_url
    FROM promotions_table
    WHERE is_active = 1
    ORDER BY saved_date_time DESC
    LIMIT 6
""")
promos = cur.fetchall()
cur.close(); db.close()

if not promos:
    print("No active promos found in DB.")
    exit(0)

PLACEHOLDER = "https://placehold.co/600x300/1a1a2e/f5a623?text=Promo"

cards = ""
for p in promos:
    img   = p.get("image_url") or PLACEHOLDER
    link  = p.get("link") or "#"
    name  = p.get("restaurant") or ""
    ptype = p.get("promo_type") or ""
    desc  = p.get("promo_details") or ""
    cards += f"""
    <td style="width:48%;vertical-align:top;padding:8px">
      <a href="{link}" style="text-decoration:none;display:block">
        <div style="background:#1e1e35;border-radius:12px;overflow:hidden;border:1px solid #2a2a45">
          <img src="{img}" alt="{name}" width="100%"
               style="display:block;width:100%;height:180px;object-fit:cover" />
          <div style="padding:16px">
            <p style="margin:0 0 4px;font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#f5a623;font-family:Arial,sans-serif">{name}</p>
            <p style="margin:0 0 8px;font-size:15px;font-weight:700;color:#ffffff;font-family:Arial,sans-serif;line-height:1.3">{ptype}</p>
            <p style="margin:0 0 14px;font-size:13px;color:#9090b0;font-family:Arial,sans-serif;line-height:1.5">{desc[:100] + ('...' if len(desc) > 100 else '')}</p>
            <span style="display:inline-block;padding:8px 16px;background:#f5a623;color:#0d0d1a;border-radius:6px;font-size:12px;font-weight:700;font-family:Arial,sans-serif">
              Voir l'offre
            </span>
          </div>
        </div>
      </a>
    </td>"""

# Group cards into rows of 2
card_list = [c for c in cards.split('<td style="width:48%') if c.strip()]
rows_html = ""
for i in range(0, len(card_list), 2):
    pair = card_list[i:i+2]
    cells = "".join('<td style="width:48%' + c for c in pair)
    if len(pair) == 1:
        cells += '<td style="width:48%;padding:8px"></td>'
    rows_html += f'<tr>{cells}</tr>'

html = f"""<!DOCTYPE html>
<html lang="fr">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#0d0d1a">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#0d0d1a;padding:32px 16px">
    <tr><td align="center">
      <table width="640" cellpadding="0" cellspacing="0" style="max-width:640px;width:100%">

        <!-- Header -->
        <tr>
          <td style="padding-bottom:32px;text-align:center">
            <p style="margin:0 0 4px;font-size:11px;letter-spacing:3px;text-transform:uppercase;color:#f5a623;font-family:Arial,sans-serif">Chicken Wings</p>
            <h1 style="margin:0;font-size:28px;font-weight:800;color:#ffffff;font-family:Arial,sans-serif">Promotions actives</h1>
            <p style="margin:8px 0 0;font-size:14px;color:#606080;font-family:Arial,sans-serif">Les meilleures offres du moment, selectionnees pour vous</p>
          </td>
        </tr>

        <!-- Cards grid -->
        <tr>
          <td>
            <table width="100%" cellpadding="0" cellspacing="0">
              {rows_html}
            </table>
          </td>
        </tr>

        <!-- Divider -->
        <tr><td style="padding:32px 0 0">
          <hr style="border:none;border-top:1px solid #1e1e35;margin:0">
        </td></tr>

        <!-- Footer -->
        <tr><td style="padding:24px 0;text-align:center">
          <p style="margin:0 0 8px;font-size:12px;color:#404060;font-family:Arial,sans-serif">
            Vous recevez cet email car vous etes abonne au Chicken Wings Promo Dashboard.
          </p>
          <a href="#" style="font-size:12px;color:#f5a623;font-family:Arial,sans-serif">Se desabonner</a>
        </td></tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""

msg = MIMEMultipart("alternative")
msg["Subject"] = "Chicken Wings — Promotions actives"
msg["From"]    = f"Chicken Wings <{MAIL_SENDER}>"
msg["To"]      = TO
msg.attach(MIMEText(html, "html"))

print(f"Connecting to {MAIL_SERVER}:{MAIL_PORT} ...")
with smtplib.SMTP_SSL(MAIL_SERVER, MAIL_PORT) as smtp:
    smtp.login(MAIL_USER, MAIL_PASS)
    smtp.sendmail(MAIL_SENDER, TO, msg.as_string())

print(f"Email sent to {TO} with {len(promos)} promos.")
