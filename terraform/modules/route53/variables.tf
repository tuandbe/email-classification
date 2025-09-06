# Route53 Module Variables

variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "domain_name" {
  description = "The domain name for the hosted zone"
  type        = string
}

variable "api_subdomain" {
  description = "The API subdomain (e.g., api.ai.demo.dev.mie.best)"
  type        = string
}

variable "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  type        = string
}

variable "alb_zone_id" {
  description = "Zone ID of the Application Load Balancer"
  type        = string
}

variable "ttl" {
  description = "The TTL for DNS records"
  type        = number
  default     = 300
}

variable "create_www_record" {
  description = "Whether to create a www CNAME record"
  type        = bool
  default     = false
}

variable "additional_records" {
  description = "Additional DNS records to create"
  type = map(object({
    type    = string
    ttl     = number
    records = list(string)
  }))
  default = {}
}

variable "mx_records" {
  description = "MX records for email"
  type        = list(string)
  default     = []
}

variable "txt_records" {
  description = "TXT records for verification"
  type = map(list(string))
  default = {}
}

variable "tags" {
  description = "Additional tags to apply to resources"
  type        = map(string)
  default     = {}
}
