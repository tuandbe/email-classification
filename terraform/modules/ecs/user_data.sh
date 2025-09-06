#!/bin/bash
# ECS EC2 Instance User Data Script

# Update system
yum update -y

# Create swap file (2GB) for better memory management
echo "Creating 2GB swap file..."
dd if=/dev/zero of=/swapfile bs=1M count=2048
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile

# Make swap permanent
echo '/swapfile none swap sw 0 0' >> /etc/fstab

# Configure swapiness for better performance
echo 'vm.swappiness=10' >> /etc/sysctl.conf
sysctl -p

# Install CloudWatch agent
yum install -y amazon-cloudwatch-agent

# Configure ECS agent
echo ECS_CLUSTER=${cluster_name} >> /etc/ecs/ecs.config
echo ECS_ENABLE_CONTAINER_METADATA=true >> /etc/ecs/ecs.config

# Start ECS agent
start ecs

# Install Docker (if not already installed)
yum install -y docker
systemctl start docker
systemctl enable docker

# Add ec2-user to docker group
usermod -a -G docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
