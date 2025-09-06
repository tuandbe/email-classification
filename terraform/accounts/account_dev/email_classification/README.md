# Email Classification Project - Terraform Configuration

## Overview

This directory contains the Terraform configuration for the Email Classification project in the development environment.

## Architecture

The infrastructure uses:
- **ALB (Application Load Balancer)**: For load balancing and SSL termination
- **ACM (AWS Certificate Manager)**: For SSL/TLS certificates
- **ECS with EC2**: For container orchestration
- **Route53**: For DNS management
- **ECR**: For container image storage

## Cost Optimization Features

The configuration is optimized for minimal costs in development:

- **Single AZ deployment**: Only one public and one private subnet
- **t4g.micro instances**: ARM-based instances with 1GB RAM (1 vCPU, 1 GB RAM + 2GB swap)
- **Minimal ECS resources**: 0.25 vCPU, 512 MB RAM per task
- **Reduced ECR retention**: Only 5 images kept, 3-day untagged cleanup
- **Disabled ECR scanning**: Saves on scanning costs
- **Single instance**: Starts with 1 EC2 instance, scales to max 2
- **ALB instead of nginx**: Reduces container complexity and management overhead

## Files

- `main.tf` - Main infrastructure configuration
- `variables.tf` - Variable definitions
- `outputs.tf` - Output values
- `versions.tf` - Provider and Terraform version requirements
- `terraform.tfvars` - Development configuration (cost-optimized)
- `terraform.tfvars.example` - Example configuration file
- `terraform.tfvars.prod` - Production configuration example

## Configuration Files

### terraform.tfvars (Development - Cost Optimized)

```hcl
# Minimal configuration for development
ecs_instance_type = "t4g.micro"   # Instance with 1GB RAM + 2GB swap
task_cpu = 256                    # Minimum CPU
task_memory = 512                 # Minimum memory
ecr_max_image_count = 5           # Keep only 5 images
ecr_scan_on_push = false          # Disable scanning
```

### terraform.tfvars.prod (Production Example)

```hcl
# Production configuration
ecs_instance_type = "t4g.medium"  # Larger instance
task_cpu = 1024                   # 1 vCPU
task_memory = 2048                # 2 GB RAM
ecr_max_image_count = 20          # Keep more images
ecr_scan_on_push = true           # Enable scanning
```

## Deployment

### Prerequisites

1. AWS CLI configured
2. Terraform >= 1.0 installed
3. Docker installed

### Steps

1. **Initialize backend** (first time only):
   ```bash
   cd terraform/accounts/account_dev/init
   terraform init
   terraform apply
   ```

2. **Deploy project**:
   ```bash
   cd terraform/accounts/account_dev/email_classification
   terraform init
   terraform plan -var-file="terraform.tfvars"
   terraform apply -var-file="terraform.tfvars"
   ```

3. **Build and push images**:
   ```bash
   # Get ECR login (replace with your actual AWS account ID and region)
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com
   
   # Build and push
   docker build -t email-classification:latest .
   docker tag email-classification:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/email-classification:latest
   docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/email-classification:latest
   ```

## Cost Estimation (Development)

### Monthly Costs (Approximate)

- **EC2 t4g.micro**: ~$7.00/month (1 instance)
- **EBS Storage**: ~$1.20/month (10 GB gp3 including swap)
- **ECR Storage**: ~$0.50/month (5 images)
- **Route53 Hosted Zone**: ~$0.50/month
- **ALB**: ~$16.20/month (Application Load Balancer)
- **ACM Certificate**: $0/month (free for AWS services)
- **Data Transfer**: ~$1.00/month
- **CloudWatch Logs**: ~$0.50/month

**Total: ~$26.90/month** for development environment

### Cost Comparison

- **With nginx container**: ~$10.70/month + container management overhead
- **With ALB + ACM**: ~$26.90/month but with better reliability and AWS-managed SSL

### Cost Optimization Tips

1. **Stop instances when not in use**:
   ```bash
   # Scale down to 0 instances
   aws ecs update-service --cluster email-classification-cluster --service email-classification-service --desired-count 0
   ```

2. **Use Spot instances** (if available):
   - Modify launch template to use Spot instances
   - Can reduce costs by up to 90%

3. **Monitor costs**:
   - Set up AWS Budget alerts
   - Use AWS Cost Explorer
   - Review monthly bills

## Monitoring

### CloudWatch Metrics

- ECS service metrics
- EC2 instance metrics
- Application logs

### Health Checks

- Application health endpoint: `https://api.example.com/health`
- ALB health checks (target group health)
- ECS service health checks
- EC2 instance health checks

## Troubleshooting

### Common Issues

1. **Instance not starting**: Check ECS task definition and container image
2. **DNS not resolving**: Verify Route53 configuration
3. **SSL certificate issues**: Check ACM certificate validation and domain ownership
4. **ALB health check failures**: Verify target group health and security groups
5. **High costs**: Review instance types, scaling policies, and ALB usage

### Useful Commands

```bash
# Check ECS service status
aws ecs describe-services --cluster email-classification-cluster --services email-classification-service

# Check EC2 instances
aws ec2 describe-instances --filters "Name=tag:Name,Values=email-classification-ecs-instance"

# Check ECR repositories
aws ecr describe-repositories

# Check ALB status
aws elbv2 describe-load-balancers --names email-classification-dev-alb

# Check target group health
aws elbv2 describe-target-health --target-group-arn <target-group-arn>

# Check ACM certificates
aws acm list-certificates

# Check Route53 records
aws route53 list-resource-record-sets --hosted-zone-id <zone-id>
```

## Security

- All resources are properly tagged
- Security groups follow least privilege
- ECR repositories have lifecycle policies
- S3 state bucket is encrypted

## Cleanup

To destroy the infrastructure:

```bash
cd terraform/accounts/account_dev/email_classification
terraform destroy -var-file="terraform.tfvars"

cd ../init
terraform destroy
```

**Warning**: This will permanently delete all resources and data!
