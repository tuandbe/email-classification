# Email Classification Project - Main Configuration
# Deploys VPC, ECS, ECR, and Route53 for the email classification service

# Data source for current AWS region
data "aws_region" "current" {}

# Data source for current AWS caller identity
data "aws_caller_identity" "current" {}

# VPC Module
module "vpc" {
  source = "../../../modules/vpc"

  project_name = var.project_name
  environment  = var.environment
  vpc_cidr     = var.vpc_cidr

  public_subnet_cidrs  = var.public_subnet_cidrs
  private_subnet_cidrs = var.private_subnet_cidrs

  tags = var.tags
}

# ECR Module
module "ecr" {
  source = "../../../modules/ecr"

  project_name     = var.project_name
  environment      = var.environment
  repository_name  = var.ecr_repository_name

  max_image_count      = var.ecr_max_image_count
  untagged_image_days  = var.ecr_untagged_image_days
  scan_on_push         = var.ecr_scan_on_push

  tags = var.tags
}

# Route53 Module (create hosted zone first)
module "route53" {
  source = "../../../modules/route53"

  project_name    = var.project_name
  environment     = var.environment
  domain_name     = var.domain_name
  api_subdomain   = var.api_subdomain
  # ALB information will be added later
  alb_dns_name    = ""
  alb_zone_id     = ""

  ttl = var.dns_ttl

  tags = var.tags
}

# ACM Module
module "acm" {
  source = "../../../modules/acm"

  project_name = var.project_name
  environment  = var.environment
  domain_name  = var.api_subdomain
  hosted_zone_id = module.route53.hosted_zone_id

  tags = var.tags

  depends_on = [module.route53]
}

# ALB Module
module "alb" {
  source = "../../../modules/alb"

  project_name = var.project_name
  environment  = var.environment

  vpc_id             = module.vpc.vpc_id
  public_subnet_ids  = module.vpc.public_subnet_ids
  certificate_arn    = module.acm.certificate_arn
  target_port        = var.container_port
  health_check_path  = var.health_check_path

  tags = var.tags

  depends_on = [module.acm]
}

# ECS Module
module "ecs" {
  source = "../../../modules/ecs"

  project_name = var.project_name
  environment  = var.environment

  vpc_id                      = module.vpc.vpc_id
  public_subnet_ids           = module.vpc.public_subnet_ids
  ecs_ec2_security_group_id   = module.vpc.ecs_ec2_security_group_id
  target_group_arn            = module.alb.target_group_arn

  instance_type     = var.ecs_instance_type
  min_capacity      = var.ecs_min_capacity
  max_capacity      = var.ecs_max_capacity
  desired_capacity  = var.ecs_desired_capacity

  container_name  = var.container_name
  container_image = var.container_image
  container_port  = var.container_port

  task_cpu    = var.task_cpu
  task_memory = var.task_memory

  container_environment = var.container_environment

  tags = var.tags

  depends_on = [module.alb]
}

# Update Route53 with ALB information
resource "aws_route53_record" "api_update" {
  zone_id = module.route53.hosted_zone_id
  name    = var.api_subdomain
  type    = "A"

  alias {
    name                   = module.alb.alb_dns_name
    zone_id                = module.alb.alb_zone_id
    evaluate_target_health = true
  }

  depends_on = [module.alb]
}

