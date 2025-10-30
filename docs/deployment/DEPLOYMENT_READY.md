# VP Investments - Deployment Guide

**Date:** October 30, 2025  
**Status:** ✅ Ready for Deployment

---

## Three Deployment Scenarios

### 1. Local Full Stack (Primary Development) ✅

**Use Case:** Active development, testing, debugging

```
Frontend (localhost:3000) ←→ Backend (localhost:8000) ←→ Database (Supabase)
```

**Documentation:** `docs/deployment/LOCAL_DEVELOPMENT.md`

**Quick Start:**
```powershell
# Terminal 1 - Backend
uvicorn backend.api.api:app --reload

# Terminal 2 - Frontend
cd frontend && npm run dev
```

**Status:** ✅ Fully configured and tested

---

### 2. Vercel Standalone Frontend (UI Testing) 🎨

**Use Case:** Frontend testing, UI/UX review, design verification

```
Frontend (vercel.app) - NO BACKEND CONNECTION
```

**Documentation:** `docs/deployment/VERCEL_SETUP.md`

**What Works:**
- ✅ Homepage and routing
- ✅ UI components and styling
- ✅ Responsive design

**What Doesn't Work:**
- ❌ Admin login (no backend)
- ❌ Data loading (no API)
- ❌ Any database operations

**Status:** ✅ Ready to deploy (frontend only)

---

### 3. Full Production Stack (Future) 🚀

**Use Case:** Production deployment, public access, full functionality

```
Frontend (Vercel) ←→ Backend (Railway) ←→ Database (Supabase)
```

**Documentation:** `docs/deployment/RAILWAY_DEPLOYMENT.md`

**Status:** 📋 Planned (not implemented yet)

---

## What Was Done

### 1. Cleanup ✅
- Moved temporary markdown files to `docs/archive/`
  - ADMIN_*.md (8 files)
  - PHASE1_*.md (2 files)
  - PIPELINE_*.md (1 file)
  - PROGRESS_*.md (1 file)
  - QUICK_START_*.md (1 file)

### 2. Configuration ✅
- Created `backend/.env.example` with all environment variables
- Updated `frontend/next.config.ts` to support environment-based API URLs
- Frontend already has `.env.example` configured

### 3. Testing ✅
- Production build tested successfully
- Build time: 9.7s
- Bundle size: 261 KB (optimized)
- All pages compile correctly

### 4. Documentation ✅
- Created `docs/deployment/VERCEL_SETUP.md` - Complete Vercel deployment guide
- Created `docs/deployment/RAILWAY_DEPLOYMENT.md` - Future reference for backend deployment

---

## Deployment Scenarios Explained

### Scenario 1: Local Full Stack ✅ (Current Setup)

**Use Case:** Active development, testing, debugging

```
Backend (localhost:8000) ←→ Frontend (localhost:3000) ←→ Database (Supabase)
```

**To Start:**
```powershell
# Terminal 1 - Backend
uvicorn backend.api.api:app --reload

# Terminal 2 - Frontend
cd frontend && npm run dev
```

**Access:** http://localhost:3000/admin/login  
**Documentation:** `docs/deployment/LOCAL_DEVELOPMENT.md`  
**Status:** ✅ Fully functional

---

### Scenario 2: Vercel Standalone 🎨 (No Backend)

**Use Case:** Frontend UI/UX testing, design review, responsive testing

```
Frontend (vercel.app) - NO BACKEND CONNECTION
```

**What Works:**
- ✅ Homepage, routing, navigation
- ✅ UI components and styling
- ✅ Responsive design

**What Doesn't:**
- ❌ Admin login (no authentication)
- ❌ Dashboard data (no API)
- ❌ Any data operations

**To Deploy:**
1. Push to GitHub
2. Go to vercel.com/new
3. Import repo, set root: `frontend`
4. Deploy (no env vars needed)

**Documentation:** `docs/deployment/VERCEL_SETUP.md`  
**Status:** ✅ Ready to deploy

---

### Scenario 3: Full Production 🚀 (Future)

**Use Case:** Production deployment with full functionality

```
Frontend (Vercel) ←→ Backend (Railway) ←→ Database (Supabase)
```

**Documentation:** `docs/deployment/RAILWAY_DEPLOYMENT.md`  
**Status:** 📋 Planned, not implemented yet

---

## Next Steps for Vercel Deployment

### Option 1: Deploy with ngrok (Easiest)

**Best for:** Quick testing, no router configuration

```powershell
# 1. Start backend
uvicorn backend.api.api:app --reload

# 2. Start ngrok tunnel (in new terminal)
ngrok http 8000
# Note the HTTPS URL: https://abc123.ngrok.io

# 3. Deploy to Vercel
# - Go to vercel.com/new
# - Import VanPete/VP-Investments
# - Set root directory: frontend
# - Add environment variable:
#   NEXT_PUBLIC_API_URL = https://abc123.ngrok.io
# - Deploy

# 4. Test at your-app.vercel.app
```

### Option 2: Deploy with Public IP (More Reliable)

**Best for:** Longer testing sessions, stable connection

```powershell
# 1. Get your public IP
(Invoke-WebRequest -Uri "https://ifconfig.me/ip").Content

# 2. Configure router port forwarding
# - Forward port 8000 to your PC's local IP
# - Enable TCP protocol

# 3. Open Windows Firewall
New-NetFirewallRule -DisplayName "FastAPI Backend" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow

# 4. Test external access (from mobile hotspot)
curl http://YOUR_PUBLIC_IP:8000/api/auth/health

# 5. Deploy to Vercel with:
#   NEXT_PUBLIC_API_URL = http://YOUR_PUBLIC_IP:8000
```

### Option 3: Deploy with Railway Backend (Most Reliable)

**Best for:** Production-like testing, persistent URL

See `docs/deployment/RAILWAY_DEPLOYMENT.md` for detailed guide.

---

## Environment Variables Needed for Vercel

When deploying to Vercel, set these in **Project Settings → Environment Variables**:

| Variable | Value | Example |
|----------|-------|---------|
| `NEXT_PUBLIC_API_URL` | Your backend URL | `http://YOUR_IP:8000` or `https://xyz.ngrok.io` |
| `NEXT_PUBLIC_ENV` | `production` | `production` |
| `NEXT_PUBLIC_ENABLE_WEBSOCKETS` | `false` | `false` |
| `NEXT_PUBLIC_DEBUG_API` | `false` | `false` |

---

## Testing Checklist

After Vercel deployment:

- [ ] Visit `https://your-app.vercel.app/admin/login`
- [ ] Login with admin/admin123
- [ ] Dashboard loads all metrics
- [ ] Runs page shows pipeline runs
- [ ] Can view run details
- [ ] Can delete a single run
- [ ] Navigation works (Dashboard ↔ Runs)
- [ ] Logout works
- [ ] Mobile view looks good

---

## Files Changed

### Modified Files
1. `frontend/next.config.ts` - Added environment-based API URL support
2. Project structure - Moved 13 temporary files to `docs/archive/`

### Created Files
1. `backend/.env.example` - Backend environment variables template
2. `docs/deployment/VERCEL_SETUP.md` - Complete Vercel deployment guide (443 lines)
3. `docs/deployment/RAILWAY_DEPLOYMENT.md` - Future Railway deployment reference (363 lines)
4. `docs/archive/` - New directory for archived documentation

---

## Important Notes

### CORS Configuration
Backend CORS is currently set to `allow_origins=["*"]` which is fine for testing. 

For production, you should restrict it to specific domains:
```python
allow_origins=[
    "http://localhost:3000",
    "https://your-app.vercel.app",
    "https://*.vercel.app",
]
```

### Security Reminders
- Default admin password is `admin123` - change for production
- JWT secret should be strong (32+ characters) - generate new for production
- Port 8000 exposed to internet (for testing only)

### Database
- Supabase database is shared between local and Vercel
- Both environments use the same data
- Be careful with deletion testing (affects both)

---

## Helpful Commands

### Local Development
```powershell
# Start backend
uvicorn backend.api.api:app --reload

# Start frontend dev server
cd frontend
npm run dev

# Build frontend for production
cd frontend
npm run build

# Start production build locally
cd frontend
npm start
```

### Testing Backend
```powershell
# Check if backend is running
curl http://localhost:8000/api/auth/health

# Test login
$response = Invoke-WebRequest -Uri http://localhost:8000/api/auth/login -Method POST -Body '{"username":"admin","password":"admin123"}' -ContentType 'application/json' -UseBasicParsing
$response.Content | ConvertFrom-Json
```

### Git Workflow
```bash
# Check status
git status --short

# Stage and commit
git add .
git commit -m "your message"

# Push to GitHub (triggers Vercel deployment)
git push origin main
```

---

## Documentation Structure

```
docs/
├── operational_guidelines.md    # Main development guide
├── README.md                     # Project overview
├── deployment/
│   ├── VERCEL_SETUP.md          # Vercel frontend deployment (NEW)
│   └── RAILWAY_DEPLOYMENT.md    # Railway backend deployment (NEW, future use)
└── archive/
    ├── ADMIN_*.md               # Admin phase documentation (13 files)
    ├── PHASE1_*.md              # Phase 1 audit reports
    ├── PIPELINE_*.md            # Pipeline improvements
    ├── PROGRESS_*.md            # Progress tracking
    └── QUICK_START_*.md         # Quick start guides
```

---

## What's Working

✅ Backend API (14 endpoints)
  - 3 auth endpoints (login, logout, verify)
  - 6 admin endpoints (runs management, deletion)
  - 5 monitoring endpoints (dashboard, metrics, health)

✅ Frontend Admin Panel
  - Login/authentication (JWT tokens)
  - Dashboard with 4 metric categories
  - Pipeline runs list (15 runs currently)
  - Run details view
  - Single run deletion (tested successfully)
  - Navigation and logout

✅ Database Integration
  - Supabase PostgreSQL (9 tables)
  - Proper foreign key relationships
  - Cascade deletion working correctly

---

## What's Next

1. **Deploy Frontend to Vercel** (whenever you're ready)
   - Follow `docs/deployment/VERCEL_SETUP.md`
   - Choose network option (ngrok recommended for first test)
   - Test thoroughly

2. **Continue Development** (ongoing)
   - Keep working locally (backend + frontend)
   - Use Vercel deployment for production-like testing
   - Push to GitHub auto-deploys to Vercel

3. **Future: Deploy Backend to Railway** (when needed)
   - See `docs/deployment/RAILWAY_DEPLOYMENT.md`
   - Migrate from local to cloud hosting
   - Update Vercel environment variables

---

## Quick Reference

| Task | Command/Location |
|------|------------------|
| Start backend | `uvicorn backend.api.api:app --reload` |
| Start frontend dev | `cd frontend && npm run dev` |
| Build frontend | `cd frontend && npm run build` |
| Vercel guide | `docs/deployment/VERCEL_SETUP.md` |
| Railway guide | `docs/deployment/RAILWAY_DEPLOYMENT.md` |
| Environment vars | `backend/.env.example`, `frontend/.env.example` |
| Local frontend | `http://localhost:3000/admin/login` |
| Local backend | `http://localhost:8000/api/auth/health` |

---

## Questions?

- **Vercel Deployment:** See `docs/deployment/VERCEL_SETUP.md`
- **Network Setup:** See VERCEL_SETUP.md → Network Setup Options
- **Backend Issues:** Check backend terminal for errors
- **Frontend Issues:** Check browser DevTools console
- **CORS Issues:** Verify backend CORS settings allow your domain

---

**Status:** ✅ Project is clean, documented, and ready for Vercel deployment whenever you're ready!
