# Email Classification Project - Outputs

# VPC Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = module.vpc.vpc_cidr_block
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = module.vpc.public_subnet_ids
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = module.vpc.private_subnet_ids
}

# ECR Outputs
output "ecr_main_repository_url" {
  description = "URL of the main ECR repository"
  value       = module.ecr.main_repository_url
}

# ECS Outputs
output "ecs_cluster_id" {
  description = "ID of the ECS cluster"
  value       = module.ecs.cluster_id
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = module.ecs.cluster_name
}

output "ecs_service_name" {
  description = "Name of the ECS service"
  value       = module.ecs.service_name
}

output "ecs_task_definition_arn" {
  description = "ARN of the task definition"
  value       = module.ecs.task_definition_arn
}

output "ecs_task_definition_family" {
  description = "Family of the task definition"
  value       = module.ecs.task_definition_family
}

# Route53 Outputs
output "hosted_zone_id" {
  description = "ID of the hosted zone"
  value       = module.route53.hosted_zone_id
}

output "hosted_zone_name_servers" {
  description = "Name servers of the hosted zone"
  value       = module.route53.hosted_zone_name_servers
}

output "api_record_fqdn" {
  description = "FQDN of the API A record"
  value       = module.route53.api_record_fqdn
}

# ALB Outputs
output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = module.alb.alb_dns_name
}

output "alb_zone_id" {
  description = "Zone ID of the Application Load Balancer"
  value       = module.alb.alb_zone_id
}

output "alb_arn" {
  description = "ARN of the Application Load Balancer"
  value       = module.alb.alb_arn
}

# ACM Outputs
output "certificate_arn" {
  description = "ARN of the SSL certificate"
  value       = module.acm.certificate_arn
}

output "certificate_status" {
  description = "Status of the SSL certificate"
  value       = module.acm.certificate_status
}

# Application Outputs
output "api_url" {
  description = "API URL"
  value       = "https://${var.api_subdomain}"
}

output "domain_name" {
  description = "Domain name"
  value       = var.domain_name
}

output "api_subdomain" {
  description = "API subdomain"
  value       = var.api_subdomain
}

# AWS Information
output "aws_region" {
  description = "AWS region"
  value       = data.aws_region.current.name
}

output "aws_account_id" {
  description = "AWS account ID"
  value       = data.aws_caller_identity.current.account_id
}
