# Email Classification Project - Development Configuration
# Cost-optimized configuration for development environment

# Project Configuration
project_name = "email-classification"
environment  = "dev"
aws_region   = "ap-southeast-1"
aws_account_id = "497046673845"

# VPC Configuration - Minimal setup for cost optimization
vpc_cidr = "10.0.0.0/16"
public_subnet_cidrs  = ["10.0.1.0/24"]  # Single public subnet to save costs
private_subnet_cidrs = ["10.0.10.0/24"] # Single private subnet to save costs

# ECR Configuration - Cost optimized
ecr_repository_name     = "email-classification"
ecr_max_image_count     = 5              # Keep only 5 images to save storage costs
ecr_untagged_image_days = 3              # Delete untagged images after 3 days
ecr_scan_on_push        = false          # Disable scanning to save costs

# ECS Configuration - Minimal resources for cost optimization
ecs_instance_type    = "t4g.micro"       # Instance type with 1GB RAM (1 vCPU, 1 GB RAM)
ecs_min_capacity     = 1                 # Minimum 1 instance
ecs_max_capacity     = 2                 # Maximum 2 instances (for auto-scaling)
ecs_desired_capacity = 1                 # Start with 1 instance

# Container Configuration - Minimal resources
container_name  = "email-classification"
container_image = "email-classification:latest"
container_port  = 8000

# Task resource allocation - Minimal for cost optimization
task_cpu    = 256   # 0.25 vCPU (minimum for ECS)
task_memory = 512   # 512 MB RAM (minimum for ECS)

# Container environment variables
container_environment = [
  {
    name  = "ENVIRONMENT"
    value = "dev"
  },
  {
    name  = "LOG_LEVEL"
    value = "INFO"
  }
]

# Route53 Configuration
domain_name   = "dev.mie.best"
api_subdomain = "api.ai.demo.dev.mie.best"
dns_ttl       = 300

# ALB Configuration
health_check_path           = "/health"
enable_deletion_protection  = false

# Tags - Cost tracking
tags = {
  Project     = "email-classification"
  Environment = "dev"
  ManagedBy   = "terraform"
  Owner       = "tuandbe"
  CostCenter  = "engineering"
  CreatedDate = "2025-09-06"
  # Add cost allocation tags
  CostAllocation = "development"
  Budget        = "low"
}
