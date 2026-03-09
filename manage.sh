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
    echo "  update      : Reconstruit l'application (après modif du code)"
    echo "  logs        : Affiche les logs en temps réel"
    echo "  setpass     : Change le mot de passe du Bunker (Nginx Auth)"
    echo "  funnel      : Active l'exposition publique via Tailscale Funnel"
    echo "  status      : Affiche l'état des services et du tunnel"
    echo "  backup      : Sauvegarde la base de données MySQL"
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
    update)
        echo -e "${GREEN}Mise à jour et reconstruction de l'application...${NC}"
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
    *)
        show_help
        ;;
esac
