# Vercel Standalone Frontend Deployment

**Status:** ✅ Ready for Deployment  
**Last Updated:** October 30, 2025  
**Purpose:** Deploy Next.js frontend to Vercel as a **standalone application** (no backend connection)

---

## Overview

This guide covers deploying the VP Investments frontend to Vercel as a **standalone application**. The frontend will be deployed without any backend connection - it's purely for frontend testing and demonstration purposes.

⚠️ **Important:** This deployment does NOT connect to the backend. The admin panel pages will not load data. This is intended for:
- Frontend UI/UX testing
- Visual design verification
- Mobile responsiveness testing
- Deployment pipeline testing

### Architecture

```
┌─────────────────────┐
│  Vercel Frontend    │
│  (Standalone)       │         ❌ NO BACKEND CONNECTION
│  Port 443 (HTTPS)   │
└─────────────────────┘
         │
         │
         ▼
    Browser Users
    (UI/UX Testing Only)
```

### Key Points

- ✅ **Frontend:** Deployed to Vercel (free tier)
- ❌ **Backend:** NOT connected (no API calls will work)
- ❌ **Database:** NOT connected
- ✅ **Purpose:** Frontend testing, UI/UX verification, design review
- ✅ **Development:** Use local setup for full functionality (see `LOCAL_DEVELOPMENT.md`)

---

## Prerequisites

1. **Vercel Account**
   - Sign up at vercel.com (free)
   - Connect your GitHub account

2. **GitHub Repository**
   - Push your code to GitHub: github.com/VanPete/VP-Investments
   - Ensure `main` branch is up to date

3. **That's It!**
   - No backend needed
   - No database needed
   - No network configuration needed

---

## What Will Work vs. What Won't

### ✅ Will Work
- Homepage loads
- Frontend routing (navigation between pages)
- UI components render correctly
- Responsive design (mobile, tablet, desktop)
- Styling and animations
- Static pages

### ❌ Won't Work
- Admin login (no backend to authenticate)
- Dashboard data loading (no API connection)
- Pipeline runs list (no database)
- Any data fetching or API calls
- Real-time features

---

## Deployment for Full Functionality

If you need the admin panel to actually work with data, you have two options:

1. **Local Development** (Recommended for now)
   - See `docs/deployment/LOCAL_DEVELOPMENT.md`
   - Both frontend and backend run locally
   - Full functionality

2. **Full Production Deployment** (Future)
   - See `docs/deployment/RAILWAY_DEPLOYMENT.md`
   - Deploy backend to Railway
   - Connect Vercel frontend to Railway backend
   - Full production setup

---

## Simple Vercel Deployment (Standalone)

## Vercel Deployment Steps (Simplified)

### Step 1: Import Project

1. Go to vercel.com/new
2. Click **"Import Git Repository"**
3. Select **VanPete/VP-Investments**
4. Click **"Import"**

### Step 2: Configure Build Settings

Vercel should auto-detect Next.js, but verify:

| Setting | Value |
|---------|-------|
| **Framework Preset** | Next.js |
| **Root Directory** | `frontend` |
| **Build Command** | `npm run build` |
| **Output Directory** | `.next` |
| **Install Command** | `npm install` |

### Step 3: Skip Environment Variables

Since this is a standalone deployment (no backend), you don't need to set any environment variables. The frontend will build successfully but API calls will fail gracefully.

**Optional:** If you want to suppress API errors in the console, you can set:

| Name | Value |
|------|-------|
| `NEXT_PUBLIC_API_URL` | `https://example.com` (dummy URL) |
| `NEXT_PUBLIC_ENV` | `production` |

### Step 4: Deploy

1. Click **"Deploy"**
2. Wait 2-3 minutes for build to complete
3. Vercel will provide a URL: `https://your-app.vercel.app`

### Step 5: Test Deployment

1. **Visit Vercel URL**
   ```
   https://your-app.vercel.app
   ```

2. **What to Test:**
   - Homepage loads correctly
   - Navigation works (click links)
   - UI renders properly
   - Responsive design (resize browser)
   - Mobile view (DevTools mobile emulator)

3. **What Won't Work (Expected):**
   - Admin login (no backend)
   - Dashboard data (no API)
   - Any data loading pages

4. **Check Browser Console:**
   - Open DevTools → Console
   - API errors are expected (no backend connection)
   - UI should still render gracefully

---

## Troubleshooting

### Issue: "Failed to load dashboard data"

**Possible Causes:**
- Backend not accessible from internet
- CORS blocking requests
- Firewall blocking port 8000

**Solutions:**
1. Test backend accessibility:
   ```powershell
   curl http://YOUR_PUBLIC_IP:8000/api/auth/health
   ```

2. Check backend CORS settings in `backend/api/api.py`:
   ```python
   allow_origins=["*"]  # Should allow all origins for testing
   ```

3. Verify firewall rule:
   ```powershell
   Get-NetFirewallRule -DisplayName "FastAPI Backend"
   ```

### Issue: "Network Error" or "CORS Policy"

**Solutions:**
1. Add Vercel domain to CORS origins (if not using `["*"]`):
   ```python
   allow_origins=[
       "http://localhost:3000",
       "https://your-app.vercel.app",
       "https://*.vercel.app",  # All preview deployments
   ]
   ```

2. Restart backend server after CORS changes

### Issue: ngrok URL expired

**Solution:**
1. Restart ngrok:
   ```powershell
   ngrok http 8000
   ```

2. Update Vercel environment variable with new URL

3. Redeploy Vercel app (Settings → Deployments → Redeploy)

### Issue: "Authentication Failed"

**Solutions:**
1. Check JWT token in browser DevTools → Application → Local Storage
2. Verify backend admin credentials match (admin/admin123)
3. Check backend logs for authentication errors:
   ```powershell
   # Check backend terminal output
   ```

---

## Development Workflow

### Recommended Workflow

1. **Local Development** (Primary)
   - Backend: `uvicorn backend.api.api:app --reload` (port 8000)
   - Frontend: `cd frontend && npm run dev` (port 3000)
   - Use `http://localhost:3000` for development

2. **Vercel Testing** (Secondary)
   - Make changes locally
   - Push to GitHub: `git push origin main`
   - Vercel auto-deploys in 2-3 minutes
   - Test at `https://your-app.vercel.app`

3. **Backend Changes**
   - Edit backend code locally
   - Backend server auto-reloads (uvicorn `--reload` flag)
   - Vercel frontend connects to local backend instantly

### Git Workflow

```bash
# Make changes
git add .
git commit -m "feat: your feature description"

# Push to trigger Vercel deployment
git push origin main

# Check deployment status
# Visit https://vercel.com/your-username/your-app/deployments
```

### Testing Checklist

After each Vercel deployment:
- [ ] Login page loads
- [ ] Authentication works
- [ ] Dashboard metrics display correctly
- [ ] Runs page shows all pipeline runs
- [ ] Single run deletion works
- [ ] Bulk deletion works (if implemented)
- [ ] Navigation works (Dashboard ↔ Runs)
- [ ] Logout works and redirects to login
- [ ] Mobile responsive layout looks good

---

## Environment Variables Reference

### Frontend (.env.local for local dev)

```bash
# Backend API URL
NEXT_PUBLIC_API_URL=http://localhost:8000

# WebSocket URL (if using real-time features)
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws

# Environment
NEXT_PUBLIC_ENV=development

# Feature Flags
NEXT_PUBLIC_ENABLE_WEBSOCKETS=true
NEXT_PUBLIC_ENABLE_PORTFOLIO=false
NEXT_PUBLIC_ENABLE_ADVANCED_CHARTS=true

# Debug Settings
NEXT_PUBLIC_DEBUG_API=true
```

### Vercel Environment Variables (Production)

```bash
# Backend API URL (CHANGE THIS based on your network setup)
NEXT_PUBLIC_API_URL=http://YOUR_PUBLIC_IP:8000  # Or ngrok/Railway URL

# WebSocket URL (optional)
NEXT_PUBLIC_WS_URL=ws://YOUR_PUBLIC_IP:8000/ws

# Environment
NEXT_PUBLIC_ENV=production

# Feature Flags
NEXT_PUBLIC_ENABLE_WEBSOCKETS=false  # Disable for now
NEXT_PUBLIC_DEBUG_API=false  # Disable debug in production
```

---

## Cost Breakdown

| Service | Plan | Cost | Notes |
|---------|------|------|-------|
| **Vercel** | Hobby | $0/month | Free tier, auto-deployments |
| **Supabase** | Free | $0/month | Database (shared) |
| **ngrok** | Free | $0/month | Free tier (URL changes on restart) |
| **ngrok** | Personal | $8/month | Permanent URL, faster speeds |
| **Public IP** | ISP | $0/month | No cost if router supports port forwarding |
| **Total** | | **$0-8/month** | Depending on network setup |

---

## Security Considerations

### For Testing/Development

- ✅ CORS set to `["*"]` is acceptable for testing
- ✅ HTTP (not HTTPS) is okay for local backend
- ✅ Exposing port 8000 temporarily is fine for testing

### Before Production Deployment

- ⚠️ Restrict CORS to specific origins
- ⚠️ Use HTTPS for backend (Let's Encrypt, Cloudflare)
- ⚠️ Add rate limiting to API endpoints
- ⚠️ Enable authentication on all routes
- ⚠️ Use environment-based secrets (not hardcoded)
- ⚠️ Deploy backend to proper hosting (Railway, Render, AWS)

---

## Next Steps

1. **Test Current Setup**
   - Ensure local backend + frontend work perfectly
   - Run `cd frontend && npm run build` successfully

2. **Choose Network Option**
   - Decide: Public IP, ngrok, or Railway
   - Set up based on chosen option

3. **Deploy to Vercel**
   - Follow Step 1-5 above
   - Test thoroughly

4. **Iterate**
   - Continue local development
   - Push to GitHub for automatic Vercel deployments
   - Test on Vercel for production-like validation

5. **Future: Full Production Deployment**
   - See `docs/deployment/RAILWAY_DEPLOYMENT.md`
   - Deploy backend to Railway/Render
   - Update Vercel environment variables to point to production backend
   - Switch to production database credentials

---

## Additional Resources

- **Vercel Documentation:** https://vercel.com/docs
- **Next.js Deployment:** https://nextjs.org/docs/deployment
- **ngrok Documentation:** https://ngrok.com/docs
- **Railway Documentation:** https://docs.railway.app

---

**Questions or Issues?**
- Check Vercel deployment logs
- Review backend terminal output
- Test API endpoints with `curl` or Postman
- Check browser DevTools Console and Network tabs
