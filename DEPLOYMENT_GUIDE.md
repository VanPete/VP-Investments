# VP INVESTMENTS - DEPLOYMENT GUIDE

## 🚀 Quick Deployment Steps

### 1. Initial Deployment (First Time)

```bash
# Ensure you're in the project root
cd "C:\Users\willi\OneDrive\Desktop\Python Projects\VP Investments"

# Add all files to git
git add .

# Commit with a meaningful message
git commit -m "Add VP Investments Frontend v1 - Production Ready"

# Push to GitHub main branch (triggers Vercel auto-deploy)
git push origin main
```

### 2. Vercel Auto-Deploy Setup

Your project is already configured for GitHub auto-deploy. When you push to `main`:

1. **GitHub receives push** → Triggers webhook to Vercel
2. **Vercel starts build**:
   - Runs `npm run prebuild` (copies results to public/)
   - Runs `npm run build` (creates optimized production build)
   - Deploys static files to CDN
3. **Site goes live** at vanpiq.com (or your custom domain)

**Build Command** (already configured in `package.json`):
```json
{
  "scripts": {
    "copy-results": "node scripts/copy-results.js",
    "prebuild": "npm run copy-results",
    "build": "next build --turbopack"
  }
}
```

### 3. Updating Pipeline Results (Ongoing)

Every time you run the Python pipeline and want to update the website:

```bash
# Step 1: Run the Python pipeline
python run_full_pipeline.py
# This automatically saves to: frontend/public/results/pipeline_results_YYYYMMDD_HHMMSS.json

# Step 2: Commit and push
git add frontend/public/results/*.json
git commit -m "Update pipeline results - $(date)"
git push origin main

# That's it! Vercel rebuilds automatically
```

### 4. Quick Update Script (Optional)

Create a PowerShell script to automate updates:

**File**: `update-and-deploy.ps1`
```powershell
# Run pipeline (automatically saves to frontend/public/results/)
Write-Host "Running pipeline..." -ForegroundColor Cyan
python run_full_pipeline.py

# Commit and push
Write-Host "Deploying to GitHub..." -ForegroundColor Cyan
git add frontend/public/results/*.json
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"
git commit -m "Update pipeline results - $timestamp"
git push origin main

Write-Host "Deployment triggered! Check Vercel dashboard." -ForegroundColor Green
```

**Usage**:
```powershell
.\update-and-deploy.ps1
```

---

## 📊 Vercel Configuration

### Build Settings
- **Framework**: Next.js
- **Build Command**: `npm run build`
- **Output Directory**: `.next` (default)
- **Install Command**: `npm install`
- **Node Version**: 18.x or higher
- **Root Directory**: `frontend`

### Environment Variables
No environment variables needed for basic deployment. All data is static.

### Custom Domain (Optional)
1. Go to Vercel dashboard → Your project → Settings → Domains
2. Add `vanpiq.com` (or your domain)
3. Follow DNS configuration instructions
4. Wait for SSL certificate (automatic)

---

## 🧪 Testing Before Deployment

### Local Production Build
Test the production build locally before deploying:

```bash
cd frontend

# Build production version
npm run build

# Start production server
npm run start

# Visit http://localhost:3000
# Test file selector, filters, methodology page
```

### Checklist
- [ ] Pipeline results are up to date
- [ ] `npm run copy-results` completed successfully
- [ ] `npm run build` completes without errors
- [ ] File selector dropdown works
- [ ] Filters work (group, factor, coverage, search)
- [ ] Methodology page loads
- [ ] Navigation links work
- [ ] Mobile responsive (test on phone or small window)

---

## 🔍 Monitoring Deployment

### Vercel Dashboard
1. Go to https://vercel.com/dashboard
2. Click on your project
3. View deployment status
4. Check build logs if errors occur

### Build Logs
If deployment fails, check:
- `npm run copy-results` output - Are results copied?
- TypeScript errors - Any type mismatches?
- Missing files - Are all dependencies installed?

---

## 🐛 Troubleshooting

### "No pipeline results found"
**Cause**: Results files not in `frontend/public/results/`
**Fix**: 
```bash
# Run the pipeline to generate results
python run_full_pipeline.py
# Results are automatically saved to frontend/public/results/

# Commit and deploy
git add frontend/public/results/*.json
git commit -m "Add pipeline results"
git push origin main
```

### File selector shows old data
**Cause**: Results not committed or outdated
**Fix**:
```bash
# Run pipeline to generate new results
python run_full_pipeline.py

# Commit and push
git add frontend/public/results/*.json
git push origin main
```

### Build fails on Vercel
**Cause**: Missing `frontend/public/results/` directory
**Fix**: Ensure the directory exists and has at least one result file:
```bash
# Run pipeline first
python run_full_pipeline.py

# Commit results
git add frontend/public/results/*.json
git commit -m "Add pipeline results"
git push origin main
```

### TypeScript errors in build
**Cause**: Type mismatches or missing types
**Fix**: Check build output, fix types locally, test with `npm run build`

---

## 📝 Deployment Checklist

### Pre-Deployment
- [ ] Run `python run_full_pipeline.py` successfully
- [ ] Run `npm run copy-results` in frontend directory
- [ ] Test local build with `npm run build`
- [ ] Test local production server with `npm start`
- [ ] Verify file selector works
- [ ] Verify all pages load (Dashboard, Methodology)

### Deployment
- [ ] Commit all changes: `git add .`
- [ ] Meaningful commit message: `git commit -m "..."`
- [ ] Push to main: `git push origin main`
- [ ] Monitor Vercel dashboard for build status

### Post-Deployment
- [ ] Visit production URL (vanpiq.com)
- [ ] Test file selector with historical dates
- [ ] Test filters (group, factor, search)
- [ ] Test methodology page
- [ ] Test on mobile device
- [ ] Verify latest data is showing

---

## 🔄 Regular Update Workflow

**Daily/Weekly** (when you update pipeline):
```bash
# 1. Run pipeline (saves directly to frontend/public/results/)
python run_full_pipeline.py

# 2. Commit and deploy
git add frontend/public/results/*.json
git commit -m "Pipeline update - $(date)"
git push origin main
```

**That's it!** Vercel handles the rest automatically.

---

## 📚 Additional Resources

- **Next.js Documentation**: https://nextjs.org/docs
- **Vercel Documentation**: https://vercel.com/docs
- **GitHub Actions** (optional automation): https://docs.github.com/actions

---

## ✅ Current Status

- ✅ Frontend build: **PASSING**
- ✅ File selector: **WORKING**
- ✅ Auto-copy script: **CONFIGURED**
- ✅ GitHub integration: **READY**
- ✅ Vercel deployment: **READY**

**Ready for production deployment!** 🚀

---

**Last Updated**: January 20, 2025
**Version**: 1.0 (Production Ready)
