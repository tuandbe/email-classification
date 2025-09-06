# ECS Module - Main Configuration
# Creates ECS cluster with EC2 instances and service

# Data source for latest Amazon Linux 2023 ECS optimized AMI (ARM64)
# Optimized for t4g instance types (AWS Graviton processors)
# Using Amazon Linux 2023 with kernel 6.1 for better performance
data "aws_ami" "ecs_optimized" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-ecs-hvm-*-arm64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Alternative: Use SSM Parameter Store for recommended AMI (more reliable)
# Uncomment the following if you prefer using SSM Parameter Store
# data "aws_ssm_parameter" "ecs_optimized_ami" {
#   name = "/aws/service/ecs/optimized-ami/amazon-linux-2023/arm64/recommended/image_id"
# }

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = merge(
    {
      Name        = "${var.project_name}-cluster"
      Environment = var.environment
    },
    var.tags
  )
}

# ECS Cluster Capacity Providers
resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name = aws_ecs_cluster.main.name

  capacity_providers = [aws_ecs_capacity_provider.main.name]

  default_capacity_provider_strategy {
    base              = 1
    weight            = 100
    capacity_provider = aws_ecs_capacity_provider.main.name
  }
}

# ECS Capacity Provider
resource "aws_ecs_capacity_provider" "main" {
  name = "${var.project_name}-capacity-provider"

  auto_scaling_group_provider {
    auto_scaling_group_arn         = aws_autoscaling_group.ecs.arn
    managed_termination_protection = "ENABLED"

    managed_scaling {
      maximum_scaling_step_size = 1000
      minimum_scaling_step_size = 1
      status                    = "ENABLED"
      target_capacity           = 100
    }
  }

  tags = merge(
    {
      Name        = "${var.project_name}-capacity-provider"
      Environment = var.environment
    },
    var.tags
  )
}

# Launch Template for ECS EC2 instances
resource "aws_launch_template" "ecs" {
  name_prefix   = "${var.project_name}-ecs-"
  image_id      = data.aws_ami.ecs_optimized.id
  instance_type = var.instance_type

  vpc_security_group_ids = [var.ecs_ec2_security_group_id]

  iam_instance_profile {
    name = aws_iam_instance_profile.ecs_ec2.name
  }

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    cluster_name = aws_ecs_cluster.main.name
  }))

  tag_specifications {
    resource_type = "instance"
    tags = merge(
      {
        Name        = "${var.project_name}-ecs-instance"
        Environment = var.environment
      },
      var.tags
    )
  }

  tags = merge(
    {
      Name        = "${var.project_name}-ecs-launch-template"
      Environment = var.environment
    },
    var.tags
  )

  lifecycle {
    create_before_destroy = true
  }
}

# Auto Scaling Group
resource "aws_autoscaling_group" "ecs" {
  name                      = "${var.project_name}-ecs-asg"
  vpc_zone_identifier       = var.public_subnet_ids
  target_group_arns         = []
  health_check_type         = "EC2"
  health_check_grace_period = 300

  min_size         = var.min_capacity
  max_size         = var.max_capacity
  desired_capacity = var.desired_capacity

  launch_template {
    id      = aws_launch_template.ecs.id
    version = "$Latest"
  }

  tag {
    key                 = "AmazonECSManaged"
    value               = true
    propagate_at_launch = true
  }

  tag {
    key                 = "Name"
    value               = "${var.project_name}-ecs-instance"
    propagate_at_launch = true
  }

  tag {
    key                 = "Environment"
    value               = var.environment
    propagate_at_launch = true
  }

  # Enable instance protection from scale in for ECS Capacity Provider
  protect_from_scale_in = true

  lifecycle {
    create_before_destroy = true
  }
}

# IAM Role for ECS EC2 instances
resource "aws_iam_role" "ecs_ec2" {
  name = "${var.project_name}-ecs-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = merge(
    {
      Name        = "${var.project_name}-ecs-ec2-role"
      Environment = var.environment
    },
    var.tags
  )
}

# IAM Instance Profile
resource "aws_iam_instance_profile" "ecs_ec2" {
  name = "${var.project_name}-ecs-ec2-instance-profile"
  role = aws_iam_role.ecs_ec2.name

  tags = merge(
    {
      Name        = "${var.project_name}-ecs-ec2-instance-profile"
      Environment = var.environment
    },
    var.tags
  )
}

# Attach ECS Instance Policy
resource "aws_iam_role_policy_attachment" "ecs_ec2" {
  role       = aws_iam_role.ecs_ec2.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

# Attach CloudWatch Agent Policy
resource "aws_iam_role_policy_attachment" "cloudwatch_agent" {
  role       = aws_iam_role.ecs_ec2.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
}

# ECS Task Definition
resource "aws_ecs_task_definition" "main" {
  family                   = "${var.project_name}-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["EC2"]
  cpu                      = var.task_cpu
  memory                   = var.task_memory

  container_definitions = jsonencode([
    {
      name  = var.container_name
      image = var.container_image
      portMappings = [
        {
          containerPort = var.container_port
          protocol      = "tcp"
        }
      ]
      environment = var.container_environment
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
          "awslogs-region"        = data.aws_region.current.name
          "awslogs-stream-prefix" = "ecs"
        }
      }
      essential = true
    }
  ])

  tags = merge(
    {
      Name        = "${var.project_name}-task-definition"
      Environment = var.environment
    },
    var.tags
  )
}

# ECS Service
resource "aws_ecs_service" "main" {
  name            = "${var.project_name}-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.main.arn
  desired_count   = var.desired_count
  launch_type     = "EC2"

  # Network configuration for awsvpc mode
  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [var.ecs_task_security_group_id]
    assign_public_ip = false
  }

  # Load balancer configuration
  load_balancer {
    target_group_arn = var.target_group_arn
    container_name   = var.container_name
    container_port   = var.container_port
  }

  # Note: Placement constraints are optional for EC2 launch type
  # ECS will automatically place tasks on available instances in the cluster

  depends_on = [aws_ecs_cluster_capacity_providers.main]

  tags = merge(
    {
      Name        = "${var.project_name}-service"
      Environment = var.environment
    },
    var.tags
  )

  lifecycle {
    ignore_changes = [desired_count]
  }
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/${var.project_name}"
  retention_in_days = var.log_retention_days

  tags = merge(
    {
      Name        = "${var.project_name}-ecs-logs"
      Environment = var.environment
    },
    var.tags
  )
}

# Data source for current AWS region
data "aws_region" "current" {}

