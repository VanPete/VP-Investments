# Railway Backend Deployment Guide

**Status:** 📋 Reference Only (Not Implemented)  
**Last Updated:** October 30, 2025  
**Purpose:** Future guide for deploying backend to Railway for full production setup

---

## Overview

This guide is for **future use** when you're ready to deploy the backend to a production hosting platform. For now, the backend remains local while the frontend is on Vercel.

### Full Production Architecture (Future)

```
┌─────────────────────┐         ┌──────────────────────┐
│  Vercel Frontend    │         │  Railway Backend     │
│  (Next.js)          │ ◄─────► │  (FastAPI)           │
│  Port 443 (HTTPS)   │   API   │  Port 443 (HTTPS)    │
└─────────────────────┘  Calls  └──────────────────────┘
         │                                │
         │                                │
         ▼                                ▼
    Browser Users              Supabase Database
                               (Production)
```

---

## Why Railway?

- ✅ **Python Support:** Native FastAPI/Uvicorn support
- ✅ **Free Tier:** $5 credit/month (enough for testing)
- ✅ **Auto-Deploy:** GitHub integration
- ✅ **Environment Variables:** Easy management
- ✅ **HTTPS Included:** Automatic SSL certificates
- ✅ **Database Support:** Can host PostgreSQL if needed
- ✅ **Logs & Monitoring:** Built-in observability

### Alternatives

- **Render.com:** Similar to Railway, free tier available
- **Fly.io:** Global edge deployment, free tier
- **Google Cloud Run:** Pay-per-use, more scalable
- **AWS ECS/Fargate:** Enterprise-grade, more complex

---

## Prerequisites

1. **Railway Account**
   - Sign up at railway.app
   - Connect GitHub account

2. **GitHub Repository**
   - Push backend code to GitHub
   - Ensure `requirements.txt` is up to date

3. **Supabase Database**
   - Production database credentials ready
   - Or use Railway PostgreSQL addon

4. **Environment Variables**
   - All secrets documented
   - See `.env.example` files

---

## Deployment Steps (Future Implementation)

### Step 1: Prepare Backend for Deployment

1. **Create `Procfile`** (Railway entry point)
   ```
   web: uvicorn backend.api.api:app --host 0.0.0.0 --port $PORT
   ```

2. **Update `requirements.txt`**
   ```bash
   pip freeze > requirements.txt
   ```

3. **Create `railway.json` (optional)**
   ```json
   {
     "build": {
       "builder": "nixpacks"
     },
     "deploy": {
       "startCommand": "uvicorn backend.api.api:app --host 0.0.0.0 --port $PORT",
       "restartPolicyType": "ON_FAILURE",
       "restartPolicyMaxRetries": 10
     }
   }
   ```

### Step 2: Create Railway Project

1. Go to railway.app/new
2. Select **"Deploy from GitHub repo"**
3. Choose **VanPete/VP-Investments**
4. Railway auto-detects Python project

### Step 3: Configure Environment Variables

Add in Railway project settings:

```bash
# Database
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_production_key

# JWT
JWT_SECRET=your_production_secret_32_chars_minimum
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=8

# Admin
ADMIN_USERNAME=admin
ADMIN_PASSWORD=your_secure_password  # Change from default!

# CORS (Update with Vercel domain)
CORS_ORIGINS=https://your-app.vercel.app,https://*.vercel.app

# External Services (if needed)
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
NEWS_API_KEY=your_news_api_key
OPENAI_API_KEY=your_openai_api_key
```

### Step 4: Deploy

1. Railway automatically deploys on push to `main`
2. Wait 3-5 minutes for build
3. Railway provides URL: `https://your-app.railway.app`

### Step 5: Update Vercel Frontend

1. Go to Vercel project settings → Environment Variables
2. Update `NEXT_PUBLIC_API_URL`:
   ```
   NEXT_PUBLIC_API_URL=https://your-app.railway.app
   ```
3. Redeploy Vercel frontend

### Step 6: Test End-to-End

1. Visit Vercel frontend: `https://your-app.vercel.app/admin/login`
2. Login with admin credentials
3. Verify dashboard loads data from Railway backend
4. Test all features (runs, deletion, navigation)

---

## Configuration Details

### CORS Configuration

Update `backend/api/api.py` for production:

```python
# Production CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-app.vercel.app",      # Production
        "https://*.vercel.app",             # Preview deployments
        "http://localhost:3000",            # Local development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Database Connection

Ensure Supabase client initialization uses environment variables:

```python
from supabase import create_client
import os

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)
```

### Health Check Endpoint

Railway uses `/` for health checks. Ensure it exists:

```python
@app.get("/")
async def health_check():
    return {"status": "healthy", "service": "VP Investments API"}
```

---

## Cost Estimation

### Railway Pricing

| Plan | Price | Compute | Notes |
|------|-------|---------|-------|
| **Trial** | $5 credit/month | Shared CPU | Free tier, credit expires |
| **Developer** | $5/month | Dedicated CPU | 500MB RAM, 1GB disk |
| **Team** | $20/month | Dedicated CPU | 2GB RAM, 5GB disk |

### Typical Backend Usage

- **Idle:** ~0.1 vCPU, 100MB RAM
- **Active:** ~0.5 vCPU, 200MB RAM
- **Estimated:** $5-10/month for light testing

### Total Production Cost

| Service | Plan | Cost |
|---------|------|------|
| Vercel | Hobby | $0/month |
| Railway | Developer | $5/month |
| Supabase | Free | $0/month |
| **Total** | | **$5/month** |

---

## Monitoring & Logs

### Railway Dashboard

- **Logs:** Real-time application logs
- **Metrics:** CPU, memory, network usage
- **Deployments:** Build history and rollback
- **Environment:** Variable management

### Endpoints to Monitor

- `GET /` - Health check (should return 200)
- `POST /api/auth/login` - Authentication
- `GET /api/monitoring/dashboard` - Dashboard metrics
- `GET /api/admin/runs` - Pipeline runs list

### Alert Setup (Future)

- Configure Railway webhooks for deployment failures
- Set up Sentry for error tracking
- Add Datadog/New Relic for APM

---

## Security Best Practices

### Before Production Deployment

1. **Change Default Credentials**
   ```bash
   ADMIN_PASSWORD=your_secure_password_here_not_admin123
   ```

2. **Restrict CORS Origins**
   - Remove `["*"]`
   - Add specific Vercel domains only

3. **Use Strong JWT Secret**
   ```bash
   JWT_SECRET=$(openssl rand -hex 32)
   ```

4. **Enable Rate Limiting**
   ```python
   from slowapi import Limiter
   from slowapi.util import get_remote_address
   
   limiter = Limiter(key_func=get_remote_address)
   app.state.limiter = limiter
   
   @app.post("/api/auth/login")
   @limiter.limit("5/minute")
   async def login(...):
       ...
   ```

5. **Add Request Validation**
   - Pydantic models for all inputs
   - Input sanitization
   - SQL injection prevention

6. **Enable HTTPS Only**
   ```python
   from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
   app.add_middleware(HTTPSRedirectMiddleware)
   ```

---

## Rollback Procedure

If deployment fails or has issues:

1. **Railway Console**
   - Go to Deployments tab
   - Click "Rollback" on previous working deployment

2. **Git Revert**
   ```bash
   git revert HEAD
   git push origin main
   ```

3. **Manual Fix**
   - Fix issue locally
   - Test thoroughly
   - Push fix to GitHub

---

## Migration Checklist

When ready to deploy to Railway:

### Pre-Deployment

- [ ] Test backend locally with production-like data
- [ ] Update `requirements.txt` with all dependencies
- [ ] Create `Procfile` for Railway
- [ ] Update CORS settings for Vercel domain
- [ ] Change default admin password
- [ ] Generate strong JWT secret
- [ ] Document all environment variables
- [ ] Test all API endpoints with Postman/curl

### Deployment

- [ ] Create Railway project
- [ ] Add all environment variables
- [ ] Deploy from GitHub
- [ ] Verify health check endpoint returns 200
- [ ] Test authentication endpoint
- [ ] Test dashboard data loading
- [ ] Test admin operations (runs, deletion)

### Post-Deployment

- [ ] Update Vercel environment variables
- [ ] Redeploy Vercel frontend
- [ ] Test end-to-end flow (Vercel → Railway → Supabase)
- [ ] Monitor Railway logs for errors
- [ ] Set up error tracking (Sentry)
- [ ] Configure backup strategy
- [ ] Document production URLs and credentials (securely)

### Ongoing Maintenance

- [ ] Monitor Railway usage and costs
- [ ] Review logs weekly for errors
- [ ] Update dependencies monthly
- [ ] Run security audits quarterly
- [ ] Test disaster recovery procedures

---

## Troubleshooting

### Issue: Build Fails on Railway

**Solutions:**
1. Check `requirements.txt` syntax
2. Ensure Python version compatibility (3.11+)
3. Review build logs for missing dependencies
4. Add `runtime.txt` with Python version:
   ```
   python-3.11.0
   ```

### Issue: App Crashes on Start

**Solutions:**
1. Check Railway logs for error messages
2. Verify all environment variables are set
3. Test Supabase connection locally
4. Ensure `Procfile` command is correct
5. Check port binding (use `$PORT` variable)

### Issue: Vercel Can't Connect to Railway

**Solutions:**
1. Verify Railway app is running (health check endpoint)
2. Check CORS settings include Vercel domain
3. Ensure `NEXT_PUBLIC_API_URL` in Vercel is correct
4. Test Railway endpoint directly:
   ```bash
   curl https://your-app.railway.app/api/auth/health
   ```

---

## Next Steps

1. **Continue Local Development**
   - Keep backend local for now
   - Use Vercel frontend for production-like testing
   - This guide serves as reference for future deployment

2. **When Ready for Production**
   - Follow this guide step-by-step
   - Test thoroughly in Railway's preview environment
   - Migrate gradually (test → staging → production)

3. **Future Enhancements**
   - Add CI/CD pipeline (GitHub Actions)
   - Set up automated testing
   - Implement blue-green deployments
   - Add load balancing for scaling

---

## Additional Resources

- **Railway Documentation:** <https://docs.railway.app>
- **FastAPI Deployment:** <https://fastapi.tiangolo.com/deployment/>
- **Uvicorn Configuration:** <https://www.uvicorn.org/deployment/>

---

**Note:** This is a reference guide. The backend remains local for now. When you're ready to deploy, return to this document and follow the steps carefully.
