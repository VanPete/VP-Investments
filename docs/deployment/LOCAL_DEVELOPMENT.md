# Local Development Setup

**Purpose:** Run both backend and frontend locally for full-stack development  
**Last Updated:** October 30, 2025  
**Status:** ✅ Primary Development Environment

---

## Overview

This is your **primary development environment** where both backend (FastAPI) and frontend (Next.js) run locally on your machine.

```
┌─────────────────────┐         ┌──────────────────────┐
│  Frontend (Local)   │         │  Backend (Local)     │
│  Next.js Dev Server │ ◄─────► │  FastAPI + Uvicorn   │
│  Port 3000          │   API   │  Port 8000           │
└─────────────────────┘  Calls  └──────────────────────┘
         │                                │
         │                                │
         ▼                                ▼
    Browser (localhost)        Supabase Database
                               (Cloud)
```

---

## Prerequisites

- ✅ Python 3.11+ installed
- ✅ Node.js 18+ installed
- ✅ Git installed
- ✅ VS Code (recommended)
- ✅ Supabase account with database created

---

## Initial Setup (One-Time)

### 1. Clone Repository

```powershell
cd "C:\Users\willi\OneDrive\Desktop\Python Projects"
git clone https://github.com/VanPete/VP-Investments.git
cd VP-Investments
```

### 2. Backend Setup

```powershell
# Install Python dependencies
pip install -r requirements.txt

# Create .env file from template
Copy-Item backend\.env.example backend\.env

# Edit backend\.env with your credentials
notepad backend\.env
```

**Required variables in `backend\.env`:**
```bash
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
JWT_SECRET=your_secret_key_32_chars
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin123
```

### 3. Frontend Setup

```powershell
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Create .env.local file from template
Copy-Item .env.example .env.local

# Edit .env.local (should already be configured for localhost)
notepad .env.local
```

**Required variables in `frontend\.env.local`:**
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_ENV=development
NEXT_PUBLIC_DEBUG_API=true
```

---

## Starting Development Servers

### Quick Start (2 Terminals Required)

**Terminal 1 - Backend Server:**
```powershell
# From project root
cd "C:\Users\willi\OneDrive\Desktop\Python Projects\VP Investments"

# Start backend with auto-reload
uvicorn backend.api.api:app --reload
```

**Expected Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using WatchFiles
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**Terminal 2 - Frontend Server:**
```powershell
# From project root
cd "C:\Users\willi\OneDrive\Desktop\Python Projects\VP Investments\frontend"

# Start frontend dev server
npm run dev
```

**Expected Output:**
```
▲ Next.js 15.5.4 (Turbopack)
- Local:        http://localhost:3000
- Environments: .env.local

✓ Starting...
✓ Ready in 2.1s
```

---

## Accessing the Application

### Main URLs

| Service | URL | Purpose |
|---------|-----|---------|
| **Frontend** | http://localhost:3000 | Main application |
| **Admin Login** | http://localhost:3000/admin/login | Admin panel login |
| **Admin Dashboard** | http://localhost:3000/admin/dashboard | Metrics dashboard |
| **Admin Runs** | http://localhost:3000/admin/runs | Pipeline runs |
| **Backend API** | http://localhost:8000 | API server |
| **API Docs** | http://localhost:8000/docs | Swagger UI |
| **API Health** | http://localhost:8000/api/auth/health | Health check |

### Default Credentials

- **Username:** admin
- **Password:** admin123

---

## Development Workflow

### Making Changes

**Backend Changes:**
1. Edit files in `backend/` directory
2. Server auto-reloads (uvicorn `--reload` flag)
3. Refresh browser to see changes
4. Check backend terminal for errors

**Frontend Changes:**
1. Edit files in `frontend/src/` directory
2. Turbopack auto-reloads instantly
3. Browser auto-refreshes (Fast Refresh)
4. Check browser console for errors

### Testing

**Test Backend API:**
```powershell
# Health check
curl http://localhost:8000/api/auth/health

# Login test
$body = @{username="admin"; password="admin123"} | ConvertTo-Json
Invoke-WebRequest -Uri http://localhost:8000/api/auth/login -Method POST -Body $body -ContentType "application/json"
```

**Test Frontend Build:**
```powershell
cd frontend
npm run build
npm start
# Visit http://localhost:3000
```

---

## Stopping Servers

**Stop Backend:**
- Press `Ctrl+C` in backend terminal

**Stop Frontend:**
- Press `Ctrl+C` in frontend terminal

**Force Stop All Node/Python Processes:**
```powershell
# List running processes
Get-Process | Where-Object {$_.ProcessName -like "*python*" -or $_.ProcessName -like "*node*"}

# Stop specific process by ID
Stop-Process -Id <PROCESS_ID> -Force
```

---

## Project Structure

```
VP Investments/
├── backend/                    # FastAPI backend
│   ├── api/                   # API endpoints
│   │   ├── admin.py          # Admin routes (runs, deletion)
│   │   ├── auth.py           # Authentication (login, JWT)
│   │   └── monitoring.py     # Monitoring (dashboard, metrics)
│   ├── core/                 # Core business logic
│   ├── integrations/         # External services
│   ├── phases/               # Pipeline phases
│   ├── storage/              # Database layer
│   ├── utils/                # Utilities
│   ├── pipeline.py           # Main pipeline
│   └── .env                  # Backend config (create from .env.example)
│
├── frontend/                  # Next.js frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── admin/        # Admin panel pages
│   │   │   │   ├── dashboard/
│   │   │   │   ├── login/
│   │   │   │   └── runs/
│   │   │   └── page.tsx      # Home page
│   │   └── components/        # React components
│   ├── .env.local            # Frontend config (create from .env.example)
│   └── package.json          # Dependencies
│
├── docs/                      # Documentation
│   ├── deployment/           # Deployment guides
│   └── operational_guidelines.md
│
├── migrations/               # Database migrations
├── config/                   # Pipeline configuration
└── requirements.txt          # Python dependencies
```

---

## Common Tasks

### Run Data Pipeline

```powershell
# From project root
python run_pipeline_and_push.py
```

### Database Migrations

```powershell
# Check database schema
python tables.py --list

# Run migration
psql $SUPABASE_URL -f migrations/014_add_sector_to_signals.sql
```

### Install New Dependencies

**Python:**
```powershell
pip install new-package
pip freeze > requirements.txt
```

**Node.js:**
```powershell
cd frontend
npm install new-package
npm install  # Update package-lock.json
```

### Update Environment Variables

**Backend:**
1. Edit `backend\.env`
2. Restart backend server (Ctrl+C, then rerun uvicorn)

**Frontend:**
1. Edit `frontend\.env.local`
2. Restart frontend server (Ctrl+C, then rerun npm run dev)

---

## Troubleshooting

### Backend Won't Start

**Issue:** "Address already in use" on port 8000

**Solution:**
```powershell
# Find process using port 8000
Get-NetTCPConnection -LocalPort 8000 | Select-Object OwningProcess
# Stop the process
Stop-Process -Id <PROCESS_ID> -Force
```

### Frontend Won't Start

**Issue:** "Port 3000 is already in use"

**Solution:**
```powershell
# Kill node processes
Get-Process node | Stop-Process -Force
```

### Database Connection Errors

**Issue:** "Could not connect to Supabase"

**Solutions:**
1. Check `backend\.env` has correct SUPABASE_URL and SUPABASE_KEY
2. Verify Supabase project is active
3. Check internet connection
4. Restart backend server

### CORS Errors

**Issue:** "CORS policy: No 'Access-Control-Allow-Origin' header"

**Solution:**
- Verify backend is running on port 8000
- Check `backend/api/api.py` has CORS middleware configured
- Restart backend server

### Authentication Fails

**Issue:** "Invalid credentials" when logging in

**Solutions:**
1. Check username/password in browser (admin/admin123)
2. Verify JWT_SECRET is set in `backend\.env`
3. Clear browser localStorage and cookies
4. Check backend terminal for auth errors

---

## Environment Variables Reference

### Backend (.env)

```bash
# Database
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key

# Authentication
JWT_SECRET=your_secret_key_minimum_32_characters
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=8

# Admin Credentials
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin123

# API
PORT=8000
CORS_ORIGINS=http://localhost:3000
```

### Frontend (.env.local)

```bash
# Backend API
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws

# Environment
NEXT_PUBLIC_ENV=development

# Feature Flags
NEXT_PUBLIC_ENABLE_WEBSOCKETS=true
NEXT_PUBLIC_ENABLE_PORTFOLIO=false
NEXT_PUBLIC_ENABLE_ADVANCED_CHARTS=true

# Debug
NEXT_PUBLIC_DEBUG_API=true
```

---

## Performance Tips

### Backend Optimization
- Use `--reload` only in development (slower startup)
- For production-like testing, remove `--reload` flag
- Enable logging to debug performance issues

### Frontend Optimization
- Use `npm run dev` for fast development (Turbopack)
- Use `npm run build` to test production build
- Clear browser cache if seeing stale data

---

## Git Workflow

```bash
# Check status
git status --short

# Pull latest changes
git pull origin main

# Create feature branch
git checkout -b feature/your-feature

# Stage changes
git add .

# Commit with message
git commit -m "feat: your feature description"

# Push to GitHub
git push origin feature/your-feature

# Merge to main (after testing)
git checkout main
git merge feature/your-feature
git push origin main
```

---

## VS Code Setup (Recommended)

### Recommended Extensions

- Python (Microsoft)
- Pylance (Microsoft)
- ESLint (Microsoft)
- Prettier - Code formatter
- Tailwind CSS IntelliSense
- GitLens

### Workspace Settings

Create `.vscode/settings.json`:
```json
{
  "python.defaultInterpreterPath": "python",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true
  },
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter"
  },
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[typescriptreact]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  }
}
```

### Launch Configuration

Create `.vscode/launch.json`:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Backend",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": [
        "backend.api.api:app",
        "--reload"
      ],
      "jinja": true
    }
  ]
}
```

---

## Next Steps

- ✅ Both servers running? → Start developing!
- ✅ Ready to deploy frontend? → See `docs/deployment/VERCEL_STANDALONE.md`
- ✅ Ready for full production? → See `docs/deployment/RAILWAY_DEPLOYMENT.md`

---

**Need Help?**
- Check backend terminal output for API errors
- Check browser DevTools console for frontend errors
- Review `docs/operational_guidelines.md` for project architecture
