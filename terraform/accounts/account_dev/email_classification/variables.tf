# Email Classification Project - Variables

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "email-classification"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}


# VPC Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.10.0/24", "10.0.20.0/24"]
}

# ECR Configuration
variable "ecr_repository_name" {
  description = "Name of the ECR repository"
  type        = string
  default     = "email-classification"
}

variable "ecr_max_image_count" {
  description = "Maximum number of images to keep in ECR"
  type        = number
  default     = 10
}

variable "ecr_untagged_image_days" {
  description = "Number of days to keep untagged images"
  type        = number
  default     = 7
}

variable "ecr_scan_on_push" {
  description = "Whether to scan images on push"
  type        = bool
  default     = true
}

# ECS Configuration
variable "ecs_instance_type" {
  description = "EC2 instance type for ECS cluster"
  type        = string
  default     = "t4g.micro"
}

variable "ecs_min_capacity" {
  description = "Minimum number of EC2 instances"
  type        = number
  default     = 1
}

variable "ecs_max_capacity" {
  description = "Maximum number of EC2 instances"
  type        = number
  default     = 3
}

variable "ecs_desired_capacity" {
  description = "Desired number of EC2 instances"
  type        = number
  default     = 1
}

# Container Configuration
variable "container_name" {
  description = "Name of the container"
  type        = string
  default     = "email-classification"
}

variable "container_image" {
  description = "Docker image for the container"
  type        = string
  default     = "email-classification:latest"
}

variable "container_port" {
  description = "Port that the container listens on"
  type        = number
  default     = 8000
}

variable "task_cpu" {
  description = "CPU units for the task (1024 = 1 vCPU)"
  type        = number
  default     = 512
}

variable "task_memory" {
  description = "Memory for the task in MB"
  type        = number
  default     = 1024
}

variable "container_environment" {
  description = "Environment variables for the container"
  type = list(object({
    name  = string
    value = string
  }))
  default = [
    {
      name  = "ENVIRONMENT"
      value = "dev"
    }
  ]
}

# Route53 Configuration
variable "domain_name" {
  description = "The domain name for the hosted zone"
  type        = string
  default     = "dev.mie.best"
}

variable "api_subdomain" {
  description = "The API subdomain"
  type        = string
  default     = "api.ai.demo.dev.mie.best"
}

variable "dns_ttl" {
  description = "TTL for DNS records"
  type        = number
  default     = 300
}

# ALB Configuration
variable "health_check_path" {
  description = "Health check path for ALB"
  type        = string
  default     = "/health"
}


# Tags
variable "tags" {
  description = "Additional tags to apply to resources"
  type        = map(string)
  default = {
    Project     = "email-classification"
    Environment = "dev"
    ManagedBy   = "terraform"
  }
}
