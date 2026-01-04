# AWS EC2 Deployment (Nginx + Gunicorn) — Production outline

## 1) EC2 instance
- Ubuntu 22.04 LTS (recommended)
- Security group: allow 22 (SSH), 80 (HTTP), 443 (HTTPS)
- Allocate Elastic IP (optional but recommended)

## 2) Install system dependencies
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv nginx git
```

## 3) App setup
```bash
git clone <your-repo>
cd heartcloud_app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: SECRET_KEY, DEBUG=0, ALLOWED_HOSTS, DATABASE_URL (Postgres recommended)
python manage.py migrate
python manage.py collectstatic --noinput
python manage.py createsuperuser
```

## 4) Gunicorn systemd service
Create `/etc/systemd/system/heartcloud.service`:
```ini
[Unit]
Description=HeartCloud Gunicorn
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/heartcloud_app
Environment="PATH=/home/ubuntu/heartcloud_app/.venv/bin"
EnvironmentFile=/home/ubuntu/heartcloud_app/.env
ExecStart=/home/ubuntu/heartcloud_app/.venv/bin/gunicorn heartcloud.wsgi:application --bind 127.0.0.1:8000 --workers 3

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now heartcloud
sudo systemctl status heartcloud
```

## 5) Nginx site config
Create `/etc/nginx/sites-available/heartcloud`:
```nginx
server {
    listen 80;
    server_name your-domain-or-ip;

    location /static/ {
        alias /home/ubuntu/heartcloud_app/staticfiles/;
    }

    location /media/ {
        alias /home/ubuntu/heartcloud_app/media/;
    }

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```
Enable:
```bash
sudo ln -s /etc/nginx/sites-available/heartcloud /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## 6) HTTPS
Use certbot:
```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## Notes
- For production, use Postgres (RDS) and S3 for media.
- Enable security settings in `settings.py` (secure cookies, HSTS, etc).
