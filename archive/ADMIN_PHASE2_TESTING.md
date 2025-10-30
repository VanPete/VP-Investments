# Admin Phase 2 Testing Guide

## Overview
This guide walks through testing the complete Admin Phase 2 implementation including JWT authentication, bulk deletion, and system monitoring dashboard.

## Prerequisites

### Backend Requirements
Ensure all dependencies are installed:
```bash
pip install python-jose[cryptography]>=3.3.0 passlib[bcrypt]>=1.7.4
```

### Frontend Requirements
Ensure frontend is running:
```bash
cd frontend
npm install
npm run dev
```

### Backend Server
Ensure FastAPI backend is running:
```bash
python -m uvicorn backend.api.api:app --reload --port 8000
```

## Test Scenarios

### 1. Authentication Flow

#### 1.1 Login Page Access
1. Navigate to `http://localhost:3000/admin/login`
2. **Expected**: Login form displays with username/password fields
3. **Verify**: Default credentials shown (admin/admin123)

#### 1.2 Valid Login
1. Enter credentials:
   - Username: `admin`
   - Password: `admin123`
2. Click "Sign In"
3. **Expected**:
   - Success toast: "Logged in successfully"
   - Redirect to `/admin/dashboard`
   - Token stored in localStorage (check DevTools → Application → Local Storage)

#### 1.3 Invalid Login
1. Enter wrong credentials
2. Click "Sign In"
3. **Expected**:
   - Error toast: "Invalid credentials"
   - Remains on login page
   - No token stored

#### 1.4 Protected Route Access
1. Clear localStorage (delete admin_token)
2. Try to navigate to `/admin/dashboard`
3. **Expected**: Automatic redirect to `/admin/login`

### 2. Dashboard Monitoring

#### 2.1 Dashboard Load
1. Login successfully
2. Navigate to `/admin/dashboard` (or auto-redirected)
3. **Expected**:
   - System Health section displays:
     - CPU Usage (percentage with color coding)
     - Memory Usage (percentage + GB used/total)
     - Disk Usage (percentage + GB used/total)
   - Pipeline Performance section displays:
     - Total Runs count
     - Success Rate percentage
     - Avg Signals count
     - Last Run timestamp
   - Factor Quality section displays:
     - Overall success rate
     - Recent runs bar chart (last 5 runs)
     - Failed factors count
   - Storage Metrics section displays:
     - Pipeline Runs count
     - Signals count
     - Analytics records count
     - Database size estimate

#### 2.2 Auto-Refresh
1. On dashboard, verify "Auto-refresh On" button is active
2. Wait 30 seconds
3. **Expected**:
   - "Last updated" timestamp updates automatically
   - Dashboard data refreshes (check "Last Run" changes if new run exists)

#### 2.3 Manual Refresh
1. Click "Refresh" button
2. **Expected**:
   - Button shows spinner icon while loading
   - "Last updated" timestamp updates immediately
   - All metrics refresh

#### 2.4 Toggle Auto-Refresh
1. Click "Auto-refresh On" button
2. **Expected**:
   - Button changes to "Auto-refresh Off"
   - Button changes to outline variant
3. Wait 30 seconds
4. **Expected**: Dashboard does NOT auto-refresh
5. Click button again
6. **Expected**: Auto-refresh re-enabled

#### 2.5 Health Status Colors
1. Check CPU/Memory/Disk cards
2. **Expected color coding**:
   - Green (< 60%): Healthy status with CheckCircle icon
   - Yellow (60-80%): Warning status with AlertCircle icon
   - Red (> 80%): Critical status with AlertCircle icon

### 3. Navigation

#### 3.1 Sidebar Navigation (Desktop)
1. Ensure browser width > 768px
2. Verify navigation bar shows:
   - VP Admin logo (left)
   - Dashboard link
   - Pipeline Runs link
   - Logout button (right)
3. Click "Dashboard"
4. **Expected**: Dashboard link highlights (blue background)
5. Click "Pipeline Runs"
6. **Expected**: 
   - Navigates to `/admin/runs`
   - Pipeline Runs link highlights

#### 3.2 Mobile Navigation
1. Resize browser to < 768px width
2. **Expected**: Navigation links collapse into hamburger menu
3. Click hamburger menu icon
4. **Expected**: Mobile menu drops down with all links
5. Click "Dashboard"
6. **Expected**:
   - Navigates to dashboard
   - Mobile menu closes automatically
7. Open menu again, click "Logout"
8. **Expected**:
   - Menu closes
   - Logout executes

#### 3.3 Logout
1. Click "Logout" button (desktop or mobile)
2. **Expected**:
   - Success toast: "Logged out successfully"
   - Redirect to `/admin/login`
   - Token removed from localStorage
3. Try to navigate to `/admin/dashboard`
4. **Expected**: Redirected back to login

### 4. Bulk Deletion

#### 4.1 Selection UI
1. Navigate to `/admin/runs`
2. **Expected**:
   - Each run card has a checkbox (Square icon when unchecked)
   - "Select All" button visible above runs list
   - No "Delete X Selected" button (nothing selected yet)

#### 4.2 Single Selection
1. Click on a run card (anywhere except existing buttons)
2. **Expected**:
   - Checkbox changes to CheckSquare icon
   - Card gets blue ring border (ring-2 ring-primary)
   - "Delete 1 Selected" button appears at top
   - Header shows "1 selected"
3. Click card again
4. **Expected**:
   - Checkbox returns to Square icon
   - Ring border disappears
   - Delete button disappears

#### 4.3 Select All
1. Click "Select All" button
2. **Expected**:
   - All run cards show CheckSquare
   - All cards have ring border
   - "Delete X Selected" button shows total count
   - Header shows "X selected"
3. Click "Select All" again (now says "Deselect All")
4. **Expected**:
   - All selections cleared
   - Delete button disappears

#### 4.4 Mixed Selection
1. Manually select 3-5 runs (click individual cards)
2. Verify count in "Delete X Selected" button matches
3. Click "Select All"
4. **Expected**: All runs now selected (not just the 3-5)
5. Click individual cards to deselect
6. **Expected**: Count decreases, button updates

#### 4.5 Bulk Delete Preview
1. Select 2-3 runs
2. Click "Delete X Selected" button
3. **Expected**:
   - Dialog opens: "Confirm Bulk Deletion"
   - Shows list of selected runs with details:
     - Run ID
     - Number of signals
     - Number of analytics records
     - Number of factors
     - Other table counts
   - Shows "Grand Total: X records across Y runs"
   - Two buttons: "Cancel" and "Delete X Runs Permanently"

#### 4.6 Cancel Bulk Delete
1. In preview dialog, click "Cancel"
2. **Expected**:
   - Dialog closes
   - Runs remain selected
   - No deletion occurs

#### 4.7 Execute Bulk Delete
1. Select 2-3 runs again
2. Click "Delete X Selected" → "Delete X Runs Permanently"
3. **Expected**:
   - Dialog shows loading state
   - Success toast: "Successfully deleted X runs"
   - Deleted runs removed from list
   - Selection cleared
   - Remaining runs still visible
   - Total runs count updates

#### 4.8 Bulk Delete with Failures
(This tests error handling - may need to simulate by deleting same runs twice)
1. Select multiple runs
2. Delete them
3. If some fail:
   - **Expected**: Toast shows "Deleted X runs (Y failed)"
   - Failed runs remain in list
   - Successfully deleted runs removed

#### 4.9 Scrollable Preview
1. Select 10+ runs (if available)
2. Open delete preview
3. **Expected**:
   - Dialog shows scrollable area (max-h-96)
   - All runs visible with scroll
   - Grand total still visible at bottom

### 5. API Authentication

#### 5.1 Valid Token
1. Login successfully
2. Open DevTools → Network tab
3. Navigate to dashboard or runs page
4. **Expected**: All API requests include:
   - Header: `Authorization: Bearer <token>`
   - Response: 200 OK

#### 5.2 Expired Token
1. Login successfully
2. Open DevTools → Application → Local Storage
3. Modify token to invalid value (e.g., add "x" at end)
4. Try to load dashboard or delete a run
5. **Expected**:
   - Response: 401 Unauthorized
   - Toast: "Session expired. Please login again."
   - Redirect to `/admin/login`

#### 5.3 Token Persistence
1. Login successfully
2. Navigate to dashboard
3. Refresh page (F5)
4. **Expected**:
   - No redirect to login
   - Dashboard loads immediately
   - Token still in localStorage

#### 5.4 Token Expiration (8 hours)
1. Login successfully
2. Wait 8+ hours (or manually set token with expired timestamp)
3. Try to access any admin page
4. **Expected**: Redirect to login due to expired token

### 6. Error Handling

#### 6.1 Network Errors
1. Stop backend server
2. Try to login or load dashboard
3. **Expected**:
   - Error toast: "Failed to fetch" or "Network error"
   - Graceful error display (not crash)
4. Restart backend
5. Click "Retry" or "Refresh"
6. **Expected**: Data loads successfully

#### 6.2 Backend Errors (500)
1. Modify backend to return 500 error (or trigger by invalid data)
2. Try to delete run or load dashboard
3. **Expected**:
   - Error toast with message
   - UI remains functional
   - Can retry action

#### 6.3 Missing Data
1. Delete all pipeline runs from database
2. Navigate to `/admin/runs`
3. **Expected**:
   - Message: "No pipeline runs found"
   - No error crash
   - Can still navigate to dashboard

### 7. UI/UX Verification

#### 7.1 Responsive Design
Test at different breakpoints:
- Mobile (< 768px): Single column, mobile menu
- Tablet (768-1024px): 2 columns for metrics cards
- Desktop (> 1024px): Full grid layouts

#### 7.2 Loading States
1. Verify loading spinners appear during:
   - Initial dashboard load
   - Bulk delete preview fetch
   - Bulk delete execution
   - Manual refresh

#### 7.3 Visual Feedback
1. Check hover states on:
   - Navigation links
   - Run cards
   - Buttons
2. Check selected state on:
   - Run cards (ring border)
   - Checkboxes (Square → CheckSquare)

#### 7.4 Toast Notifications
Verify toasts appear for:
- ✅ Login success
- ❌ Login failure
- ✅ Logout success
- ✅ Bulk delete success
- ❌ Bulk delete errors
- ❌ Session expiration
- ❌ Network errors

### 8. Security Verification

#### 8.1 Role-Based Access
1. Verify only admin role can access admin pages
2. Check backend logs for authorization checks
3. **Expected**: All admin endpoints check `get_current_admin_user()`

#### 8.2 Password Security
1. Check backend USERS_DB
2. **Expected**: Passwords stored as bcrypt hashes (not plaintext)
3. Verify login endpoint uses `verify_password()`

#### 8.3 Token Security
1. Check token in localStorage
2. **Expected**: JWT format with 3 parts (header.payload.signature)
3. Decode token (jwt.io)
4. **Expected**: Contains username, role, expiration

## Backend API Testing

### Direct API Tests (using curl or Postman)

#### Login
```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
```
**Expected**: `{"access_token": "<token>", "token_type": "bearer", "user": {...}}`

#### Get Current User
```bash
curl http://localhost:8000/api/auth/me \
  -H "Authorization: Bearer <token>"
```
**Expected**: `{"username": "admin", "role": "admin", ...}`

#### System Health
```bash
curl http://localhost:8000/api/monitoring/health \
  -H "Authorization: Bearer <token>"
```
**Expected**: CPU, memory, disk metrics

#### Dashboard Overview
```bash
curl http://localhost:8000/api/monitoring/dashboard \
  -H "Authorization: Bearer <token>"
```
**Expected**: Combined metrics from all endpoints

#### Bulk Delete Preview
```bash
curl -X POST http://localhost:8000/api/admin/runs/bulk-delete \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"run_ids": ["<run_id_1>", "<run_id_2>"], "confirm": false}'
```
**Expected**: Preview with record counts per run

#### Bulk Delete Execute
```bash
curl -X POST http://localhost:8000/api/admin/runs/bulk-delete \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"run_ids": ["<run_id_1>", "<run_id_2>"], "confirm": true}'
```
**Expected**: Deletion results with total count

## Performance Testing

### Dashboard Load Time
1. Open DevTools → Network tab
2. Load dashboard
3. **Target**: < 2 seconds for initial load
4. **Verify**: API calls complete within 500ms each

### Bulk Delete Performance
1. Select 10 runs
2. Execute bulk delete
3. **Target**: < 5 seconds for completion
4. **Verify**: Individual cascade deletes complete efficiently

### Auto-Refresh Impact
1. Enable auto-refresh
2. Monitor Network tab for 2 minutes
3. **Expected**:
   - Requests every 30 seconds
   - No memory leaks
   - UI remains responsive

## Regression Testing

After testing Phase 2 features, verify Phase 1 functionality still works:

1. **Single Run Deletion**:
   - Individual delete button on each run
   - Preview shows counts correctly
   - Deletion executes successfully

2. **Run Details**:
   - Can view individual run details
   - Ticker list displays correctly
   - Factor monitoring data visible

3. **Filtering/Sorting**:
   - Search runs by ticker
   - Sort by date/success
   - Pagination works

## Known Issues & Edge Cases

### Edge Case 1: Empty Database
- **Scenario**: No pipeline runs exist
- **Expected**: "No pipeline runs found" message
- **Should NOT**: Crash or show errors

### Edge Case 2: Selecting All Runs
- **Scenario**: Select all runs on page (e.g., 20+)
- **Expected**: Preview dialog scrollable, bulk delete works
- **Should NOT**: Timeout or freeze UI

### Edge Case 3: Concurrent Deletions
- **Scenario**: Two admins delete same run simultaneously
- **Expected**: One succeeds, one gets 404 error
- **Should NOT**: Database inconsistency

### Edge Case 4: Long-Running Deletion
- **Scenario**: Delete run with 10,000+ signals
- **Expected**: Loading state shown, eventually completes
- **Should NOT**: Timeout without feedback

## Success Criteria

✅ All Phase 2 features working:
- JWT authentication with 8-hour token expiration
- Bulk selection and deletion (multiple runs at once)
- Monitoring dashboard with auto-refresh
- Shared admin layout with navigation

✅ Security properly implemented:
- All admin endpoints protected
- Passwords hashed with bcrypt
- Tokens validated on every request
- Role-based access control enforced

✅ UX meets expectations:
- Loading states for all async operations
- Toast notifications for all actions
- Responsive design (mobile + desktop)
- Error handling with user feedback

✅ Performance acceptable:
- Dashboard loads < 2s
- Bulk operations complete < 5s
- Auto-refresh doesn't degrade performance
- No memory leaks on long sessions

## Troubleshooting

### Issue: "Session expired" immediately after login
**Cause**: Token expiration time too short or clock skew
**Fix**: Check backend `ACCESS_TOKEN_EXPIRE_MINUTES` setting

### Issue: Bulk delete fails with "run not found"
**Cause**: Run already deleted or invalid run_id
**Fix**: Verify run exists before deletion, handle 404 gracefully

### Issue: Dashboard shows "Failed to load"
**Cause**: Backend monitoring endpoints not accessible
**Fix**: Check backend logs, verify psutil is installed

### Issue: Can't login with admin/admin123
**Cause**: USERS_DB not initialized or password hash mismatch
**Fix**: Check backend auth.py USERS_DB initialization

### Issue: Token not persisting across page refresh
**Cause**: localStorage not working in private/incognito mode
**Fix**: Use regular browser window or session storage

## Next Steps

After successful testing:
1. ✅ Document any bugs found
2. ✅ Update ADMIN_PHASE2_COMPLETE.md with test results
3. ✅ Consider additional features:
   - User management (add/edit/delete admins)
   - Audit log viewer
   - Export data functionality
   - Advanced filtering/search
4. ✅ Deploy to production environment
5. ✅ Monitor real-world usage and performance
