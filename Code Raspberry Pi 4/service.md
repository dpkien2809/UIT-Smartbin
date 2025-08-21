# Run Python Project as a Systemd Service

This guide shows how to create and run a Python project as a **systemd service** on Linux.

## 1. Create the Service File

Create a new service file:

```bash
sudo nano /etc/systemd/system/myproject.service
```

Add the following content (update paths accordingly):

```ini
[Unit]
Description=My Python Project Service
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/myproject
ExecStart=/usr/bin/python3 /home/ubuntu/myproject/main.py
Restart=always
RestartSec=20

[Install]
WantedBy=multi-user.target
```

## 2. Reload Systemd

```bash
sudo systemctl daemon-reload
```

## 3. Start the Service

```bash
sudo systemctl start myproject.service
```

## 4. Check Service Status

```bash
sudo systemctl status myproject.service
```

## 5. Enable Service on Boot

```bash
sudo systemctl enable myproject.service
```

## 6. Stop and Disable Service

```bash
sudo systemctl stop myproject.service
sudo systemctl disable myproject.service
```

## 7. View Logs

```bash
journalctl -u myproject.service -f
sudo systemctl status myproject.service
```
