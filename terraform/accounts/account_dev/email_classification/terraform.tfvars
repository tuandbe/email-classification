# Email Classification Project - Development Configuration
# Cost-optimized configuration for development environment

# Project Configuration
project_name = "email-classification"
environment  = "dev"

# VPC Configuration - Minimal setup for cost optimization
vpc_cidr             = "10.0.0.0/16"
public_subnet_cidrs  = ["10.0.1.0/24", "10.0.2.0/24"]   # Two public subnets for ALB (required)
private_subnet_cidrs = ["10.0.10.0/24", "10.0.20.0/24"] # Two private subnets for consistency

# ECR Configuration - Cost optimized
ecr_repository_name     = "email-classification"
ecr_max_image_count     = 5     # Keep only 5 images to save storage costs
ecr_untagged_image_days = 3     # Delete untagged images after 3 days
ecr_scan_on_push        = false # Disable scanning to save costs

# ECS Configuration - Optimized for FastAPI application
ecs_instance_type    = "t4g.small" # Instance type with 2GB RAM (1 vCPU, 2 GB RAM)
ecs_min_capacity     = 1           # Minimum 1 instance
ecs_max_capacity     = 2           # Maximum 2 instances (for auto-scaling)
ecs_desired_capacity = 1           # Start with 1 instance

# Container Configuration - Optimized for FastAPI
container_name = "email-classification"
container_port = 8000

# Task resource allocation - Adequate for FastAPI + ML model
task_cpu    = 512  # 0.5 vCPU (sufficient for FastAPI)
task_memory = 1024 # 1GB RAM (sufficient for FastAPI + ML model)

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
health_check_path = "/health"

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
  Budget         = "low"
}
