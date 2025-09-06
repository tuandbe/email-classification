# Terraform Infrastructure for Email Classification Project

This directory contains the Terraform configuration for deploying the email classification service on AWS.

## Architecture

The infrastructure includes:

- **VPC**: Virtual Private Cloud with public and private subnets
- **ECS**: Elastic Container Service with EC2 instances (t4g.small)
- **ECR**: Elastic Container Registry for Docker images
- **Route53**: DNS management for domain `example.com`

## Project Structure

```
terraform/
├── .terraform-version      # Terraform version
├── .tflint.hcl            # TFLint configuration
├── .trivyignore           # Trivy ignore file
├── modules/               # Reusable Terraform modules
│   ├── vpc/              # VPC module
│   ├── ecs/              # ECS module
│   ├── ecr/              # ECR module
│   └── route53/          # Route53 module
└── accounts/
    └── account_dev/      # Development environment
        ├── init/         # Backend initialization
        └── email_classification/  # Project configuration
```

## Prerequisites

1. AWS CLI configured with appropriate credentials
2. Terraform >= 1.0 installed
3. Docker installed (for building images)

## Deployment Steps

### 1. Initialize Backend

```bash
cd terraform/accounts/account_dev/init
terraform init
terraform plan
terraform apply
```

### 2. Deploy Infrastructure

```bash
cd terraform/accounts/account_dev/email_classification
terraform init
terraform plan
terraform apply
```

### 3. Build and Push Docker Images

```bash
# Build main application image
docker build -t email-classification:latest .

# Build nginx image
docker build -f Dockerfile.nginx -t email-classification-nginx:latest .

# Get ECR login token (replace with your actual AWS account ID and region)
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com

# Tag and push images
docker tag email-classification:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/email-classification:latest
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/email-classification:latest
```

## Configuration

### Environment Variables

Update the following variables in `terraform/accounts/account_dev/email_classification/variables.tf`:

- `container_image`: ECR repository URL for your Docker image
- `domain_name`: Your domain name
- `api_subdomain`: API subdomain

### Backend Configuration

Update the S3 bucket name in `versions.tf` files:

```hcl
backend "s3" {
  bucket = "your-terraform-state-bucket"
  region = "us-east-1"
  key    = "terraform/accounts/account_dev/email_classification/terraform.tfstate"
}
```

## Outputs

After deployment, you'll get:

- API URL: `https://api.example.com`
- ECR repository URLs
- ECS cluster information
- Route53 hosted zone details

## Security

- All resources are properly tagged
- S3 bucket for state is encrypted
- ECR repositories have lifecycle policies
- Security groups follow least privilege principle

## Monitoring

- CloudWatch logs are enabled for ECS tasks
- Container insights are enabled for the ECS cluster
- Health checks are configured for the application

## Cleanup

To destroy the infrastructure:

```bash
cd terraform/accounts/account_dev/email_classification
terraform destroy

cd ../init
terraform destroy
```

## Troubleshooting

1. **DNS not resolving**: Check Route53 hosted zone name servers
2. **SSL certificate issues**: Verify domain ownership and DNS propagation
3. **ECS tasks not starting**: Check CloudWatch logs and task definition
4. **ECR push failures**: Verify AWS credentials and repository permissions
