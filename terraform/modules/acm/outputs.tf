# ACM Module - Outputs

output "certificate_arn" {
  description = "ARN of the SSL certificate"
  value       = aws_acm_certificate.main.arn
}

output "certificate_domain_name" {
  description = "Domain name of the certificate"
  value       = aws_acm_certificate.main.domain_name
}

output "certificate_status" {
  description = "Status of the certificate"
  value       = aws_acm_certificate.main.status
}

output "certificate_validation_status" {
  description = "Status of the certificate validation"
  value       = aws_acm_certificate_validation.main.id
}

output "validation_records" {
  description = "DNS validation records"
  value       = aws_route53_record.validation
}
