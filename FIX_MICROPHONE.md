# Fix Microphone Permission Issue

## Problem Found
Your browser has the **Microphone permission set to "Block"** for `http://0.0.0.0:8000`.

## Solution

### Step 1: Enable Microphone Permission

In the Chrome settings page you're viewing:

1. Find the **"Microphone"** row in the Permissions section
2. Click on the dropdown that says **"Block"**
3. Change it to either:
   - **"Ask"** (recommended) - Browser will ask permission each time
   - **"Allow"** - Always allow microphone access for this site

### Step 2: Use localhost Instead of 0.0.0.0

**Important:** Use `localhost` instead of `0.0.0.0` for better compatibility:

1. Close the current tab
2. Open a new tab
3. Go to: `http://localhost:8000/ui/index.html`
4. Chrome will ask for microphone permission (if you set it to "Ask")
5. Click **"Allow"** when prompted

### Step 3: Test

1. Open browser console (F12)
2. Click the microphone button
3. You should see:
   - "Requesting microphone access..."
   - Browser permission prompt appears
   - "Microphone access granted"
   - "Recording started"

## Alternative: Reset All Permissions

If you want to start fresh:

1. In the site settings page, click **"Reset permissions"** button (top right)
2. This will reset all permissions to default
3. Then use `http://localhost:8000/ui/index.html`
4. Browser will ask for microphone permission

## Quick Fix Summary

1. ✅ Change Microphone from "Block" to "Ask" or "Allow"
2. ✅ Use `http://localhost:8000/ui/index.html` instead of `0.0.0.0`
3. ✅ Click microphone button
4. ✅ Allow permission when browser asks

That's it! The microphone should work now.
