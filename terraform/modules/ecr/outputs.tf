# ECR Module Outputs

output "main_repository_url" {
  description = "URL of the main ECR repository"
  value       = aws_ecr_repository.main.repository_url
}

output "main_repository_arn" {
  description = "ARN of the main ECR repository"
  value       = aws_ecr_repository.main.arn
}

output "main_repository_name" {
  description = "Name of the main ECR repository"
  value       = aws_ecr_repository.main.name
}

output "main_repository_registry_id" {
  description = "Registry ID of the main ECR repository"
  value       = aws_ecr_repository.main.registry_id
}
