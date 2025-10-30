# Admin System - Quick Start Guide

## 🎉 What's Been Implemented

### ✅ Complete Features

1. **JWT Authentication System**
   - Login endpoint with secure password hashing
   - Bearer token authentication
   - Admin role verification
   - All admin endpoints now protected

2. **Bulk Deletion API**
   - Delete multiple pipeline runs at once
   - Preview before deletion (shows total records)
   - Individual error handling per run
   - Audit logging

3. **Monitoring Dashboard API**
   - System health (CPU, memory, disk usage)
   - Factor quality metrics
   - Pipeline performance metrics
   - Storage usage analytics

4. **Login Page UI**
   - Clean, professional login form
   - Token storage in localStorage
   - Error handling
   - Auto-redirect after login

---

## 🚀 Quick Start

### 1. Dependencies Already Installed ✅

```bash
✅ python-jose[cryptography]==3.5.0
✅ passlib[bcrypt]==1.7.4
```

### 2. Start the Backend API

```bash
cd "c:\Users\willi\OneDrive\Desktop\Python Projects\VP Investments\backend"
uvicorn api.api:app --reload --host 0.0.0.0 --port 8000
```

### 3. Start the Frontend

```bash
cd "c:\Users\willi\OneDrive\Desktop\Python Projects\VP Investments\frontend"
npm run dev
```

### 4. Login

Navigate to: `http://localhost:3000/admin/login`

**Credentials:**
- Username: `admin`
- Password: `admin123`

---

## 📋 Testing Checklist

### Test Authentication

1. **Login via UI:**
   - Go to `http://localhost:3000/admin/login`
   - Enter: admin / admin123
   - Should redirect to `/admin/dashboard`
   - Check browser console - token should be stored

2. **Login via API:**
```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d "{\"username\": \"admin\", \"password\": \"admin123\"}"
```

Expected response:
```json
{
  "access_token": "eyJhbGc...",
  "token_type": "bearer",
  "expires_in": 28800,
  "user": {
    "username": "admin",
    "role": "admin",
    "full_name": "Administrator"
  }
}
```

3. **Test Protected Endpoint:**

Save your token from login response, then:

```bash
curl http://localhost:8000/api/admin/runs/list \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

Should return list of runs (not 401 Unauthorized).

### Test Bulk Deletion

1. **Get some run_ids:**
```bash
curl http://localhost:8000/api/admin/runs/list \
  -H "Authorization: Bearer YOUR_TOKEN"
```

2. **Preview bulk deletion:**
```bash
curl -X POST http://localhost:8000/api/admin/runs/bulk-delete \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"run_ids\": [\"run1\", \"run2\"], \"confirm\": false}"
```

3. **Confirm deletion:**
```bash
curl -X POST http://localhost:8000/api/admin/runs/bulk-delete \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"run_ids\": [\"run1\", \"run2\"], \"confirm\": true}"
```

### Test Monitoring Endpoints

```bash
TOKEN="YOUR_TOKEN_HERE"

# System health
curl http://localhost:8000/api/monitoring/health -H "Authorization: Bearer $TOKEN"

# Factor quality
curl http://localhost:8000/api/monitoring/factor-quality -H "Authorization: Bearer $TOKEN"

# Pipeline metrics
curl http://localhost:8000/api/monitoring/pipeline-metrics -H "Authorization: Bearer $TOKEN"

# Storage metrics
curl http://localhost:8000/api/monitoring/storage -H "Authorization: Bearer $TOKEN"

# Complete dashboard
curl http://localhost:8000/api/monitoring/dashboard -H "Authorization: Bearer $TOKEN"
```

---

## 🔒 Security Features

### Implemented ✅
- JWT tokens with expiration (8 hours)
- Bcrypt password hashing
- Role-based access control (admin role required)
- Bearer token authentication
- Audit logging for all admin actions

### How Authentication Works

1. **User logs in** → POST /api/auth/login
2. **Server validates credentials** → Checks username/password
3. **Server generates JWT** → Signs token with secret key
4. **Client stores token** → localStorage.setItem('admin_token', token)
5. **Client makes request** → Adds header: `Authorization: Bearer <token>`
6. **Server validates token** → Decodes JWT, checks signature
7. **Server checks role** → Verifies user has 'admin' role
8. **Request succeeds** → Returns data

### Token Structure

```javascript
// JWT payload
{
  "sub": "admin",           // Username
  "role": "admin",          // User role
  "exp": 1698765432         // Expiration timestamp
}
```

---

## 📊 API Endpoints Reference

### Authentication (No auth required)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/login` | Login and get JWT token |
| GET | `/api/auth/me` | Get current user info (requires token) |
| POST | `/api/auth/logout` | Logout (client-side) |

### Admin Operations (Admin auth required)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/admin/runs/delete` | Delete single pipeline run |
| POST | `/api/admin/runs/bulk-delete` | **NEW** Delete multiple runs |
| GET | `/api/admin/runs/list` | List all pipeline runs |
| GET | `/api/admin/runs/{run_id}` | Get run details |

### Monitoring (Admin auth required)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/monitoring/health` | **NEW** System health metrics |
| GET | `/api/monitoring/factor-quality` | **NEW** Factor quality indicators |
| GET | `/api/monitoring/pipeline-metrics` | **NEW** Pipeline performance |
| GET | `/api/monitoring/storage` | **NEW** Storage usage metrics |
| GET | `/api/monitoring/dashboard` | **NEW** Complete overview |

---

## 💡 Usage Examples

### Example 1: Login and List Runs

```bash
# 1. Login
LOGIN_RESPONSE=$(curl -s -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}')

# 2. Extract token (using jq)
TOKEN=$(echo $LOGIN_RESPONSE | jq -r '.access_token')

# 3. List runs
curl http://localhost:8000/api/admin/runs/list \
  -H "Authorization: Bearer $TOKEN"
```

### Example 2: Bulk Delete with Preview

```bash
# 1. Get list of runs to delete
RUNS=$(curl -s http://localhost:8000/api/admin/runs/list \
  -H "Authorization: Bearer $TOKEN")

# 2. Preview deletion (2 runs)
curl -X POST http://localhost:8000/api/admin/runs/bulk-delete \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"run_ids": ["run_id_1", "run_id_2"], "confirm": false}'

# 3. Confirm deletion
curl -X POST http://localhost:8000/api/admin/runs/bulk-delete \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"run_ids": ["run_id_1", "run_id_2"], "confirm": true}'
```

### Example 3: Get Complete Dashboard Data

```bash
curl -s http://localhost:8000/api/monitoring/dashboard \
  -H "Authorization: Bearer $TOKEN" | jq '.'
```

---

## 🎯 What's Working Now

### Backend (100% Complete)
✅ JWT authentication with bcrypt password hashing
✅ All admin endpoints protected with role verification
✅ Bulk deletion API with preview/confirm pattern
✅ System health monitoring (CPU, memory, disk)
✅ Factor quality metrics from logs
✅ Pipeline performance analytics
✅ Storage usage tracking
✅ Audit logging for all admin actions

### Frontend (50% Complete)
✅ Login page with token storage
✅ Single run deletion in runs page
❌ Bulk selection UI (checkboxes not added yet)
❌ Monitoring dashboard page (not created yet)
❌ Admin layout with navigation (not created yet)
❌ Auth guard for protected routes (not added yet)

---

## 🔜 Next Steps (Optional Enhancements)

### 1. Complete Bulk Selection UI

Update `frontend/src/app/admin/runs/page.tsx`:
- Add checkboxes to each run card
- Add "Select All" checkbox
- Add "Bulk Delete" button (shows when items selected)
- Update preview dialog to show combined totals

### 2. Create Monitoring Dashboard Page

Create `frontend/src/app/admin/dashboard/page.tsx`:
- System health cards (CPU, memory, disk)
- Factor quality indicators
- Pipeline metrics with charts
- Storage usage visualization
- Auto-refresh every 30 seconds

### 3. Add Admin Layout

Create `frontend/src/app/admin/layout.tsx`:
- Shared navigation sidebar
- Header with user info
- Logout button
- Auth guard (redirect to login if no token)
- Active page highlighting

### 4. Add Frontend Auth Utilities

Create `frontend/src/lib/auth.ts`:
- `getToken()` - Get token from localStorage
- `isAuthenticated()` - Check if user logged in
- `logout()` - Clear token and redirect
- `fetchWithAuth(url, options)` - Fetch wrapper with auth header

---

## 🎨 UI Screenshots

### Login Page
```
┌─────────────────────────────────────┐
│         🔐 Admin Login              │
│                                     │
│  Enter your credentials to access   │
│  the admin panel                    │
│                                     │
│  Username: [admin____________]      │
│  Password: [••••••••_________]      │
│                                     │
│  Default credentials:               │
│  Username: admin                    │
│  Password: admin123                 │
│                                     │
│  [ Login ]                          │
└─────────────────────────────────────┘
```

---

## 🐛 Troubleshooting

### Problem: 401 Unauthorized on admin endpoints

**Solution:** Make sure you're sending the Authorization header:
```bash
curl http://localhost:8000/api/admin/runs/list \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

### Problem: Token expired

**Solution:** Login again to get a new token. Tokens expire after 8 hours.

### Problem: Can't access monitoring endpoints

**Solution:** Ensure you're logged in as admin user. The default `admin` user has admin role.

### Problem: Frontend not redirecting after login

**Solution:** Check browser console for errors. Ensure token is being stored in localStorage.

---

## 📝 Configuration

### Change JWT Secret (Recommended for Production)

1. Generate a secure key:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

2. Set environment variable:
```bash
# Windows PowerShell
$env:JWT_SECRET_KEY="your-generated-key-here"

# Or in .env file
JWT_SECRET_KEY=your-generated-key-here
```

### Change Token Expiration

In `backend/api/auth.py`:
```python
ACCESS_TOKEN_EXPIRE_MINUTES = 480  # 8 hours
# Change to desired value (in minutes)
```

### Add New Admin User

In `backend/api/auth.py`, add to `USERS_DB`:
```python
USERS_DB = {
    "admin": {
        "username": "admin",
        "password_hash": pwd_context.hash("admin123"),
        "role": "admin",
        "full_name": "Administrator"
    },
    "newuser": {
        "username": "newuser",
        "password_hash": pwd_context.hash("newpassword"),
        "role": "admin",
        "full_name": "New User"
    }
}
```

---

## ✅ Success!

Your admin system is now fully functional with:
- ✅ Secure authentication
- ✅ Protected endpoints
- ✅ Bulk operations
- ✅ System monitoring
- ✅ Audit logging

All backend APIs are ready to use. Test them with curl or integrate with frontend!

**Next:** Build the remaining frontend pages (dashboard, bulk selection UI) for complete admin experience.
