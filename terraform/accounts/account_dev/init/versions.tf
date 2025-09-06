# Init Backend Configuration - Account Dev
# This file sets up the S3 backend for state storage

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # backend "s3" {
  #   bucket       = "tfstate-email-classification-dev"  # Replace with your S3 bucket name
  #   region       = "ap-southeast-1"                    # Replace with your region
  #   encrypt      = true
  #   key          = "terraform/accounts/account_dev/init/terraform.tfstate"
  # }
}

# Default tags for all AWS resources in this account
provider "aws" {
  region = "ap-southeast-1"
  
  default_tags {
    tags = {
      Environment = "dev"
      Project     = "email-classification"
      Owner       = "tuandbe"
      ManagedBy   = "terraform"
      CostCenter  = "engineering"
      CreatedDate = "2025-09-06"
    }
  }
}
