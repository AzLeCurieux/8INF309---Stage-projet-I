#!/bin/bash

# Gestionnaire simplifié pour l'infrastructure Bunker + Tailscale
# Usage: ./manage.sh [command]

# Couleurs pour le terminal
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

function show_help() {
    echo -e "${BLUE}=== Promo Dashboard Toolkit ===${NC}"
    echo "Usage: ./manage.sh [option]"
    echo ""
    echo "Options disponibles :"
    echo "  start       : Lance toute l'infrastructure (Docker)"
    echo "  stop        : Arrête tous les services"
    echo "  restart     : Redémarre proprement tous les services"
    echo "  pull        : Récupère le code depuis GitHub (Git pull)"
    echo "  update      : Reconstruit et redémarre l'application (sans Git pull)"
    echo "  logs        : Affiche les logs en temps réel"
    echo "  setpass     : Change le mot de passe du Bunker (Nginx Auth)"
    echo "  funnel      : Active l'exposition publique via Tailscale Funnel"
    echo "  status      : Affiche l'état des services et du tunnel"
    echo "  backup      : Sauvegarde la base de données MySQL
  clearpromos : Vide les promotions uniquement (conserve les restaurants)
  cleardb     : Vide toute la base de données (promotions + restaurants)"
}

case "$1" in
    start)
        echo -e "${GREEN}Démarrage de l'infrastructure...${NC}"
        sudo docker compose up -d
        ;;
    stop)
        echo -e "${RED}Arrêt de l'infrastructure...${NC}"
        sudo docker compose down
        ;;
    restart)
        echo -e "${BLUE}Redémarrage en cours...${NC}"
        sudo docker compose restart
        ;;
    pull)
        echo -e "${BLUE}Récupération du code depuis GitHub...${NC}"
        git pull origin main
        ;;
    update)
        echo -e "${GREEN}Reconstruction de l'application...${NC}"
        sudo docker compose up -d --build app
        ;;
    logs)
        sudo docker compose logs -f
        ;;
    status)
        echo -e "${BLUE}--- État de Docker ---${NC}"
        sudo docker compose ps
        echo -e "\n${BLUE}--- État de Tailscale ---${NC}"
        tailscale serve status
        ;;
    setpass)
        read -p "Entrez le nouveau mot de passe pour 'admin' : " newpass
        if [ -n "$newpass" ]; then
            HASH=$(openssl passwd -1 "$newpass")
            echo "admin:$HASH" > htpasswd
            sudo docker compose restart gatekeeper
            echo -e "${GREEN}Mot de passe mis à jour avec succès !${NC}"
        else
            echo -e "${RED}Erreur : Mot de passe vide.${NC}"
        fi
        ;;
    funnel)
        echo -e "${GREEN}Activation de Tailscale Funnel sur le port 8080...${NC}"
        tailscale funnel 8080
        ;;
    backup)
        DATE=$(date +%Y%m%d_%H%M%S)
        FILE="backup_promos_$DATE.sql"
        echo -e "${GREEN}Sauvegarde de la base de données vers $FILE...${NC}"
        # On récupère le nom du conteneur DB dynamiquement
        DB_CONTAINER=$(sudo docker compose ps -q db)
        sudo docker exec $DB_CONTAINER /usr/bin/mysqldump -u root -p1234 promotions_db > "$FILE"
        echo "Terminé."
        ;;
    clearpromos)
        echo -e "${RED}Attention : cette opération va supprimer toutes les promotions (les restaurants sont conservés).${NC}"
        read -p "Confirmer ? (oui/non) : " confirm
        if [ "$confirm" = "oui" ]; then
            DB_CONTAINER=$(sudo docker compose ps -q db)
            sudo docker exec $DB_CONTAINER /usr/bin/mysql -u root -p1234 promotions_db \
                -e "DELETE FROM promotions_table;"
            echo -e "${GREEN}Promotions supprimées.${NC}"
        else
            echo "Annulé."
        fi
        ;;
    cleardb)
        echo -e "${RED}Attention : cette opération va vider toute la base de données (promotions + restaurants).${NC}"
        read -p "Confirmer ? (oui/non) : " confirm
        if [ "$confirm" = "oui" ]; then
            DB_CONTAINER=$(sudo docker compose ps -q db)
            sudo docker exec $DB_CONTAINER /usr/bin/mysql -u root -p1234 promotions_db \
                -e "DELETE FROM promotions_table; DELETE FROM restaurants;"
            echo -e "${GREEN}Base de données vidée.${NC}"
        else
            echo "Annulé."
        fi
        ;;
    *)
        show_help
        ;;
esac
