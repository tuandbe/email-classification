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
AWS_REGION="ap-southeast-1"
AWS_PROFILE="${AWS_PROFILE:-default}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-}"
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
    log_info "Checking AWS credentials with profile: $AWS_PROFILE"
    if ! aws sts get-caller-identity --profile $AWS_PROFILE &> /dev/null; then
        log_error "AWS credentials not configured for profile '$AWS_PROFILE'. Please run 'aws configure --profile $AWS_PROFILE' first."
        exit 1
    fi
    
    # Check AWS Account ID
    log_info "Checking AWS Account ID..."
    if [ -z "$AWS_ACCOUNT_ID" ]; then
        log_error "AWS_ACCOUNT_ID environment variable is not set. Please set it before running the script."
        exit 1
    fi
    
    # Get actual AWS Account ID from AWS CLI
    ACTUAL_ACCOUNT_ID=$(aws sts get-caller-identity --profile $AWS_PROFILE --query Account --output text 2>/dev/null)
    if [ -z "$ACTUAL_ACCOUNT_ID" ]; then
        log_error "Failed to get AWS Account ID from AWS CLI."
        exit 1
    fi
    
    # Compare Account IDs
    if [ "$AWS_ACCOUNT_ID" != "$ACTUAL_ACCOUNT_ID" ]; then
        log_error "AWS Account ID mismatch!"
        log_error "Expected: $AWS_ACCOUNT_ID"
        log_error "Actual: $ACTUAL_ACCOUNT_ID"
        log_error "Please set the correct AWS_ACCOUNT_ID environment variable."
        exit 1
    fi
    
    log_info "AWS Account ID verified: $AWS_ACCOUNT_ID"
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
    
    # Use AWS Account ID from environment variable
    ECR_REGISTRY="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
    ECR_REPOSITORY_URI="$ECR_REGISTRY/$PROJECT_NAME"
    
    # Login to ECR
    log_info "Logging in to ECR..."
    aws ecr get-login-password --region $AWS_REGION --profile $AWS_PROFILE | docker login --username AWS --password-stdin $ECR_REGISTRY
    
    # Generate timestamp tag for unique image version
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    
    # Build main application image
    log_info "Building main application image with timestamp: $TIMESTAMP"
    docker build -t $PROJECT_NAME:$TIMESTAMP .
    docker build -t $PROJECT_NAME:latest .
    
    # Tag image for ECR
    log_info "Tagging image for ECR..."
    docker tag $PROJECT_NAME:$TIMESTAMP $ECR_REPOSITORY_URI:$TIMESTAMP
    docker tag $PROJECT_NAME:latest $ECR_REPOSITORY_URI:latest
    
    # Push image to ECR
    log_info "Pushing image to ECR..."
    docker push $ECR_REPOSITORY_URI:$TIMESTAMP
    docker push $ECR_REPOSITORY_URI:latest
    
    log_info "Image pushed with tags: $TIMESTAMP and latest"
    
    log_info "Docker images built and pushed successfully!"
}

force_update_ecs_service() {
    log_info "Force updating ECS service to use new image..."
    
    # Get cluster and service names from Terraform outputs
    CLUSTER_NAME=$(cd "$PROJECT_DIR" && terraform output -raw ecs_cluster_name 2>/dev/null || echo "email-classification-cluster")
    SERVICE_NAME=$(cd "$PROJECT_DIR" && terraform output -raw ecs_service_name 2>/dev/null || echo "email-classification-service")
    
    if [ -z "$CLUSTER_NAME" ] || [ -z "$SERVICE_NAME" ]; then
        log_error "Could not determine cluster or service name. Please check Terraform outputs."
        return 1
    fi
    
    log_info "Updating ECS service: $SERVICE_NAME in cluster: $CLUSTER_NAME"
    
    # Force new deployment
    aws ecs update-service \
        --cluster "$CLUSTER_NAME" \
        --service "$SERVICE_NAME" \
        --force-new-deployment \
        --region "$AWS_REGION" \
        --profile "$AWS_PROFILE" \
        --query 'service.serviceName' \
        --output text
    
    if [ $? -eq 0 ]; then
        log_info "ECS service update initiated successfully!"
        log_info "Waiting for service to stabilize..."
        
        # Wait for service to stabilize
        aws ecs wait services-stable \
            --cluster "$CLUSTER_NAME" \
            --services "$SERVICE_NAME" \
            --region "$AWS_REGION" \
            --profile "$AWS_PROFILE"
        
        if [ $? -eq 0 ]; then
            log_info "ECS service is now stable and running with new image!"
        else
            log_warning "ECS service update completed but may not be fully stable yet."
        fi
    else
        log_error "Failed to update ECS service"
        return 1
    fi
}
show_outputs() {
    log_info "Getting deployment outputs..."
    
    cd "$PROJECT_DIR"
    
    # Get outputs
    API_URL=$(terraform output -raw api_url 2>/dev/null || echo "Not available")
    DOMAIN_NAME=$(terraform output -raw domain_name 2>/dev/null || echo "Not available")
    ECR_MAIN_URL=$(terraform output -raw ecr_main_repository_url 2>/dev/null || echo "Not available")
    
    cd - > /dev/null
    
    echo ""
    log_info "=== DEPLOYMENT SUMMARY ==="
    echo "API URL: $API_URL"
    echo "Domain: $DOMAIN_NAME"
    echo "ECR Main Repository: $ECR_MAIN_URL"
    echo ""
    log_info "Deployment completed successfully!"
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
        force_update_ecs_service
        ;;
    "update-ecs")
        check_prerequisites
        force_update_ecs_service
        ;;
    *)
        echo "Usage: $0 {deploy|destroy|init-only|project-only|images-only|update-ecs}"
        echo ""
        echo "Commands:"
        echo "  deploy       - Deploy complete infrastructure (default)"
        echo "  destroy      - Destroy all infrastructure"
        echo "  init-only    - Deploy only init infrastructure"
        echo "  project-only - Deploy only project infrastructure"
        echo "  images-only  - Build and push Docker images only"
        echo "  update-ecs   - Force update ECS service to use latest image"
        exit 1
        ;;
esac
