# Route53 Module - Main Configuration
# Creates hosted zone and DNS records

# Hosted Zone
resource "aws_route53_zone" "main" {
  name = var.domain_name

  tags = {
    Name = "${var.project_name}-hosted-zone"
  }
}

# A Record for API subdomain (ALB alias) - only create if ALB info is provided
resource "aws_route53_record" "api" {
  count = var.alb_dns_name != "" ? 1 : 0

  zone_id = aws_route53_zone.main.zone_id
  name    = var.api_subdomain
  type    = "A"

  alias {
    name                   = var.alb_dns_name
    zone_id                = var.alb_zone_id
    evaluate_target_health = true
  }
}

# CNAME Record for www (optional)
resource "aws_route53_record" "www" {
  count = var.create_www_record ? 1 : 0

  zone_id = aws_route53_zone.main.zone_id
  name    = "www.${var.domain_name}"
  type    = "CNAME"
  ttl     = var.ttl

  records = [var.api_subdomain]
}

# NS Records (automatically created by AWS)
# These are output for reference

# Optional: Create additional A records
resource "aws_route53_record" "additional" {
  for_each = var.additional_records

  zone_id = aws_route53_zone.main.zone_id
  name    = each.key
  type    = each.value.type
  ttl     = each.value.ttl

  records = each.value.records
}

# Optional: Create MX records for email
resource "aws_route53_record" "mx" {
  count = length(var.mx_records) > 0 ? 1 : 0

  zone_id = aws_route53_zone.main.zone_id
  name    = var.domain_name
  type    = "MX"
  ttl     = var.ttl

  records = var.mx_records
}

# Optional: Create TXT records for verification
resource "aws_route53_record" "txt" {
  for_each = var.txt_records

  zone_id = aws_route53_zone.main.zone_id
  name    = each.key
  type    = "TXT"
  ttl     = var.ttl

  records = each.value
}
