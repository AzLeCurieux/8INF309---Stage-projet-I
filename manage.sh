#!/bin/bash

# Gestionnaire du Promo Dashboard
# Usage: ./manage.sh [command]

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

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
    echo "  status      : Affiche l'état des services"
    echo "  backup      : Sauvegarde la base de données MySQL"
    echo "  clearpromos : Vide les promotions uniquement (conserve les restaurants)"
    echo "  cleardb     : Vide toute la base de données (promotions + restaurants)"
    echo "  makeadmin   : Donne le rôle admin à un utilisateur par email"
}

case "$1" in
    start)
        echo -e "${GREEN}Démarrage de l'infrastructure...${NC}"
        docker compose up -d
        ;;
    stop)
        echo -e "${RED}Arrêt de l'infrastructure...${NC}"
        docker compose down
        ;;
    restart)
        echo -e "${BLUE}Redémarrage en cours...${NC}"
        docker compose restart
        ;;
    pull)
        echo -e "${BLUE}Récupération du code depuis GitHub...${NC}"
        git pull origin main
        ;;
    update)
        echo -e "${GREEN}Reconstruction de l'application...${NC}"
        docker compose up -d --build app
        ;;
    logs)
        docker compose logs -f
        ;;
    status)
        echo -e "${BLUE}--- État des services ---${NC}"
        docker compose ps
        ;;
    backup)
        DATE=$(date +%Y%m%d_%H%M%S)
        FILE="backup_promos_$DATE.sql"
        echo -e "${GREEN}Sauvegarde vers $FILE...${NC}"
        DB_CONTAINER=$(docker compose ps -q db)
        docker exec $DB_CONTAINER /usr/bin/mysqldump -u root -p1234 promotions_db > "$FILE"
        echo "Terminé : $FILE"
        ;;
    clearpromos)
        echo -e "${RED}Attention : suppression de toutes les promotions (restaurants conservés).${NC}"
        read -p "Confirmer ? (oui/non) : " confirm
        if [ "$confirm" = "oui" ]; then
            DB_CONTAINER=$(docker compose ps -q db)
            docker exec $DB_CONTAINER /usr/bin/mysql -u root -p1234 promotions_db \
                -e "DELETE FROM promotions_table;"
            echo -e "${GREEN}Promotions supprimées.${NC}"
        else
            echo "Annulé."
        fi
        ;;
    cleardb)
        echo -e "${RED}Attention : suppression complète de la base (promotions + restaurants).${NC}"
        read -p "Confirmer ? (oui/non) : " confirm
        if [ "$confirm" = "oui" ]; then
            DB_CONTAINER=$(docker compose ps -q db)
            docker exec $DB_CONTAINER /usr/bin/mysql -u root -p1234 promotions_db \
                -e "DELETE FROM promotions_table; DELETE FROM restaurants;"
            echo -e "${GREEN}Base de données vidée.${NC}"
        else
            echo "Annulé."
        fi
        ;;
    makeadmin)
        read -p "Email de l'utilisateur à promouvoir admin : " email
        if [ -n "$email" ]; then
            DB_CONTAINER=$(docker compose ps -q db)
            RESULT=$(docker exec $DB_CONTAINER /usr/bin/mysql -u root -p1234 promotions_db \
                -se "UPDATE users SET role='admin' WHERE email='$email'; SELECT ROW_COUNT();")
            if [ "$RESULT" = "1" ]; then
                echo -e "${GREEN}✓ $email est maintenant admin.${NC}"
            else
                echo -e "${RED}Aucun utilisateur trouvé avec l'email : $email${NC}"
            fi
        else
            echo -e "${RED}Erreur : email vide.${NC}"
        fi
        ;;
    *)
        show_help
        ;;
esac
