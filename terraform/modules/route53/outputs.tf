# Route53 Module Outputs

output "hosted_zone_id" {
  description = "ID of the hosted zone"
  value       = aws_route53_zone.main.zone_id
}

output "hosted_zone_name" {
  description = "Name of the hosted zone"
  value       = aws_route53_zone.main.name
}

output "hosted_zone_name_servers" {
  description = "Name servers of the hosted zone"
  value       = aws_route53_zone.main.name_servers
}

output "api_record_name" {
  description = "Name of the API A record"
  value       = length(aws_route53_record.api) > 0 ? aws_route53_record.api[0].name : null
}

output "api_record_fqdn" {
  description = "FQDN of the API A record"
  value       = length(aws_route53_record.api) > 0 ? aws_route53_record.api[0].fqdn : null
}

output "www_record_name" {
  description = "Name of the www CNAME record"
  value       = var.create_www_record ? aws_route53_record.www[0].name : null
}

output "www_record_fqdn" {
  description = "FQDN of the www CNAME record"
  value       = var.create_www_record ? aws_route53_record.www[0].fqdn : null
}

output "domain_name" {
  description = "The domain name"
  value       = var.domain_name
}

output "api_subdomain" {
  description = "The API subdomain"
  value       = var.api_subdomain
}
