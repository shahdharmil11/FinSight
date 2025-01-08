sudo apt-get update
sudo apt-get install -y python3 python3-pip git

for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done

# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian \
  bookworm stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update


sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin


sudo mkdir -p /home/FinSight
sudo chmod -R 777 /home/FinSight
cd /home/FinSight
git clone https://github.com/aditya-prayaga/FinSight.git
cd /home/FinSight/FinSight

# CREATE USER ID in Env
echo -e "AIRFLOW_UID=$(id -u)" > .env

sudo docker compose up airflow-init
sudo docker-compose up