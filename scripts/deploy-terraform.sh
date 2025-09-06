#!/bin/bash
# Terraform Deployment Script for Email Classification Project

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="email-classification"
ENVIRONMENT="dev"
AWS_REGION="ap-northeast-1"
TERRAFORM_DIR="terraform"
INIT_DIR="$TERRAFORM_DIR/accounts/account_dev/init"
PROJECT_DIR="$TERRAFORM_DIR/accounts/account_dev/email_classification"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if AWS CLI is installed
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check if Terraform is installed
    if ! command -v terraform &> /dev/null; then
        log_error "Terraform is not installed. Please install it first."
        exit 1
    fi
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured. Please run 'aws configure' first."
        exit 1
    fi
    
    log_info "All prerequisites met!"
}

deploy_init() {
    log_info "Deploying init infrastructure..."
    
    cd "$INIT_DIR"
    
    # Initialize Terraform
    terraform init
    
    # Plan deployment
    log_info "Planning init deployment..."
    terraform plan -out=init.tfplan
    
    # Apply deployment
    log_info "Applying init deployment..."
    terraform apply init.tfplan
    
    # Clean up plan file
    rm -f init.tfplan
    
    cd - > /dev/null
    
    log_info "Init infrastructure deployed successfully!"
}

deploy_project() {
    log_info "Deploying project infrastructure..."
    
    cd "$PROJECT_DIR"
    
    # Initialize Terraform
    terraform init
    
    # Plan deployment
    log_info "Planning project deployment..."
    terraform plan -out=project.tfplan
    
    # Apply deployment
    log_info "Applying project deployment..."
    terraform apply project.tfplan
    
    # Clean up plan file
    rm -f project.tfplan
    
    cd - > /dev/null
    
    log_info "Project infrastructure deployed successfully!"
}

build_and_push_images() {
    log_info "Building and pushing Docker images..."
    
    # Get AWS account ID
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    ECR_REGISTRY="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
    
    # Login to ECR
    log_info "Logging in to ECR..."
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY
    
    # Build main application image
    log_info "Building main application image..."
    docker build -t $PROJECT_NAME:latest .
    docker tag $PROJECT_NAME:latest $ECR_REGISTRY/$PROJECT_NAME:latest
    docker push $ECR_REGISTRY/$PROJECT_NAME:latest
    
    # Build nginx image
    log_info "Building nginx image..."
    docker build -f Dockerfile.nginx -t $PROJECT_NAME-nginx:latest .
    docker tag $PROJECT_NAME-nginx:latest $ECR_REGISTRY/$PROJECT_NAME-nginx:latest
    docker push $ECR_REGISTRY/$PROJECT_NAME-nginx:latest
    
    log_info "Docker images built and pushed successfully!"
}

show_outputs() {
    log_info "Getting deployment outputs..."
    
    cd "$PROJECT_DIR"
    
    # Get outputs
    API_URL=$(terraform output -raw api_url 2>/dev/null || echo "Not available")
    DOMAIN_NAME=$(terraform output -raw domain_name 2>/dev/null || echo "Not available")
    ECR_MAIN_URL=$(terraform output -raw ecr_main_repository_url 2>/dev/null || echo "Not available")
    ECR_NGINX_URL=$(terraform output -raw ecr_nginx_repository_url 2>/dev/null || echo "Not available")
    
    cd - > /dev/null
    
    echo ""
    log_info "=== DEPLOYMENT SUMMARY ==="
    echo "API URL: $API_URL"
    echo "Domain: $DOMAIN_NAME"
    echo "ECR Main Repository: $ECR_MAIN_URL"
    echo "ECR Nginx Repository: $ECR_NGINX_URL"
    echo ""
    log_info "Deployment completed successfully!"
    log_warn "Note: It may take a few minutes for the SSL certificate to be generated and the service to be fully available."
}

destroy_infrastructure() {
    log_warn "This will destroy all infrastructure. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        log_info "Destroying project infrastructure..."
        cd "$PROJECT_DIR"
        terraform destroy -auto-approve
        cd - > /dev/null
        
        log_info "Destroying init infrastructure..."
        cd "$INIT_DIR"
        terraform destroy -auto-approve
        cd - > /dev/null
        
        log_info "Infrastructure destroyed successfully!"
    else
        log_info "Destruction cancelled."
    fi
}

# Main script
case "${1:-deploy}" in
    "deploy")
        check_prerequisites
        deploy_init
        deploy_project
        build_and_push_images
        show_outputs
        ;;
    "destroy")
        destroy_infrastructure
        ;;
    "init-only")
        check_prerequisites
        deploy_init
        ;;
    "project-only")
        check_prerequisites
        deploy_project
        ;;
    "images-only")
        check_prerequisites
        build_and_push_images
        ;;
    *)
        echo "Usage: $0 {deploy|destroy|init-only|project-only|images-only}"
        echo ""
        echo "Commands:"
        echo "  deploy       - Deploy complete infrastructure (default)"
        echo "  destroy      - Destroy all infrastructure"
        echo "  init-only    - Deploy only init infrastructure"
        echo "  project-only - Deploy only project infrastructure"
        echo "  images-only  - Build and push Docker images only"
        exit 1
        ;;
esac
