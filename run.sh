#!/bin/bash
# Usage: bash run.sh
# Prérequis: MySQL local sur 127.0.0.1:3306 avec DB promotions_db
#            (les colonnes grade/savings_estimate/category doivent exister)

VENV_PYTHON="/home/as/ProjetUqacStage/maindirectory/notebooks_and_scripts/.venv_linux/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Variables d'environnement (remplacent le .env si besoin)
export DB_HOST="127.0.0.1"
export DB_PORT="3306"
export DB_USER="root"
export DB_PASSWORD="1234"
export DB_NAME="promotions_db"
export SKIP_EMBEDDING_INIT="1"  # Mettre à 0 pour activer les embeddings (plus lent)
export SCRAPE_INTERVAL_HOURS="6"  # Scraping automatique toutes les 6 heures

echo "========================================"
echo "  Promo Dashboard – démarrage"
echo "  http://localhost:5000"
echo "========================================"
echo "Python : $VENV_PYTHON"
echo "Script : $SCRIPT_DIR/server.py"
echo "DB     : $DB_HOST:$DB_PORT/$DB_NAME"
echo "Scheduler: toutes les ${SCRAPE_INTERVAL_HOURS}h"
echo "----------------------------------------"

exec "$VENV_PYTHON" "$SCRIPT_DIR/server.py"
