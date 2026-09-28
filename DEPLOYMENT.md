# ChargeGuard Deployment Guide

This guide covers deploying ChargeGuard using Docker Compose for development, staging, and production environments.

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- Git

## Quick Start (Local Development)

1. **Clone the repository:**
```bash
git clone https://github.com/Aru-0504/chargeguard.git
cd chargeguard
```

2. **Copy environment template:**
```bash
cp .env.example .env
```

3. **Start all services:**
```bash
docker-compose up -d
```

4. **Access the application:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- PostgreSQL: localhost:5432

5. **View logs:**
```bash
docker-compose logs -f
```

## Environment Variables

### Root `.env` (for Docker Compose)

```env
# Database
POSTGRES_USER=chargeguard
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=chargeguard

# OpenAI (optional - for AI evidence generation)
OPENAI_API_KEY=your_openai_api_key

# CORS (comma-separated list of allowed origins)
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com

# Frontend API URL
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Backend Environment Variables

```env
DATABASE_URL=postgresql://user:password@host/dbname?sslmode=require
OPENAI_API_KEY=your_openai_api_key_here
CORS_ORIGINS=http://localhost:3000,http://localhost:3001
```

### Frontend Environment Variables

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Production Deployment

### Option 1: Docker Compose (Simple)

1. **Prepare production environment:**
```bash
cp .env.example .env.production
# Edit .env.production with production values
```

2. **Use production compose file:**
```bash
docker-compose --env-file .env.production up -d
```

3. **Configure reverse proxy (nginx example):**
```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Option 2: Cloud Platform Deployment

#### Deploy to Railway

1. Connect your GitHub repository to Railway
2. Railway will detect the Dockerfile and deploy automatically
3. Set environment variables in Railway dashboard:
   - `DATABASE_URL` (Railway provides PostgreSQL)
   - `OPENAI_API_KEY`
   - `CORS_ORIGINS`

#### Deploy to Render

1. Create a `render.yaml` file:
```yaml
services:
  - type: web
    name: chargeguard-backend
    env: docker
    dockerContext: ./backend
    dockerfilePath: Dockerfile
    envVars:
      - key: DATABASE_URL
        fromDatabase:
          name: chargeguard-db
          property: connectionString
      - key: OPENAI_API_KEY
        sync: false
      - key: CORS_ORIGINS
        value: https://your-app.onrender.com

  - type: web
    name: chargeguard-frontend
    env: docker
    dockerContext: ./frontend
    dockerfilePath: Dockerfile
    envVars:
      - key: NEXT_PUBLIC_API_URL
        value: https://chargeguard-backend.onrender.com

databases:
  - name: chargeguard-db
    databaseName: chargeguard
```

2. Connect repository to Render
3. Deploy

#### Deploy to AWS (ECS)

1. **Build and push images to ECR:**
```bash
# Backend
docker build -t chargeguard-backend ./backend
docker tag chargeguard-backend:latest <aws-account-id>.dkr.ecr.<region>.amazonaws.com/chargeguard-backend:latest
docker push <aws-account-id>.dkr.ecr.<region>.amazonaws.com/chargeguard-backend:latest

# Frontend
docker build -t chargeguard-frontend ./frontend
docker tag chargeguard-frontend:latest <aws-account-id>.dkr.ecr.<region>.amazonaws.com/chargeguard-frontend:latest
docker push <aws-account-id>.dkr.ecr.<region>.amazonaws.com/chargeguard-frontend:latest
```

2. **Create ECS task definitions** using the pushed images
3. **Configure load balancer** for routing
4. **Set up RDS PostgreSQL** for database

## Database Management

### Backup Database
```bash
docker-compose exec postgres pg_dump -U chargeguard chargeguard > backup.sql
```

### Restore Database
```bash
docker-compose exec -T postgres psql -U chargeguard chargeguard < backup.sql
```

### Access Database
```bash
docker-compose exec postgres psql -U chargeguard chargeguard
```

## Model Management

The backend requires `chargeback_model.pkl` and `metrics.json` files. For production:

1. **Train a model** using `backend/retrain_model.py`
2. **Copy the files** to `backend/app/` before building Docker image
3. **Or use volume mounting** in docker-compose.yml:
```yaml
volumes:
  - ./backend/app/chargeback_model.pkl:/app/app/chargeback_model.pkl
  - ./backend/app/metrics.json:/app/app/metrics.json
```

## Monitoring

### Health Checks

- Backend health: `http://your-domain/health`
- Docker Compose health: `docker-compose ps`

### Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f postgres
```

## Troubleshooting

### Backend fails to start

1. Check if model files exist:
```bash
ls -la backend/app/chargeback_model.pkl
ls -la backend/app/metrics.json
```

2. Check database connection:
```bash
docker-compose logs backend | grep -i database
```

3. Verify environment variables:
```bash
docker-compose config
```

### Frontend can't connect to backend

1. Check CORS configuration in backend `.env`
2. Verify `NEXT_PUBLIC_API_URL` in frontend `.env`
3. Check if backend is healthy:
```bash
curl http://localhost:8000/health
```

### Database connection issues

1. Check PostgreSQL is running:
```bash
docker-compose ps postgres
```

2. Test connection:
```bash
docker-compose exec postgres pg_isready -U chargeguard
```

3. Check database logs:
```bash
docker-compose logs postgres
```

## Security Checklist

- [ ] Change default PostgreSQL passwords
- [ ] Use strong `OPENAI_API_KEY` and keep it secret
- [ ] Configure CORS for production domains only
- [ ] Enable SSL/TLS for database connections
- [ ] Use HTTPS for all API endpoints
- [ ] Set up proper secrets management (AWS Secrets Manager, etc.)
- [ ] Enable rate limiting on API endpoints
- [ ] Implement authentication/authorization
- [ ] Regular security updates for dependencies

## Scaling

### Horizontal Scaling

For backend scaling, use a load balancer and multiple instances:

```yaml
# In docker-compose.yml
backend:
  deploy:
    replicas: 3
```

Or use Kubernetes/Helm charts for production scaling.

### Database Scaling

- Use managed PostgreSQL (RDS, Neon, etc.)
- Enable connection pooling (PgBouncer)
- Consider read replicas for read-heavy workloads

## Cost Optimization

- Use spot instances for non-critical workloads
- Enable auto-scaling based on traffic
- Optimize database queries
- Use CDN for static assets
- Monitor and optimize resource usage

## Support

For issues or questions:
- GitHub Issues: https://github.com/Aru-0504/chargeguard/issues
- Documentation: See README.md
