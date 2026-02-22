# Fix Microphone Permission - Step by Step

## Quick Fix

### Method 1: Browser Address Bar (Easiest)

1. **Look at the address bar** - you should see a lock icon 🔒 or microphone icon 🎤
2. **Click the icon**
3. **Find "Microphone"** in the dropdown
4. **Change it to "Allow"**
5. **Refresh the page** (F5)

### Method 2: Chrome Settings

1. **Click the lock icon** in address bar
2. **Click "Site settings"**
3. **Find "Microphone"**
4. **Change from "Block" to "Ask" or "Allow"**
5. **Refresh the page**

### Method 3: Chrome Settings Page

1. **Open**: `chrome://settings/content/microphone`
2. **Find**: `localhost:8000` in the list
3. **Set to**: "Allow"
4. **Refresh the page**

## Verify It's Fixed

After allowing permission:
1. **Refresh the page** (F5 or Ctrl+R)
2. **Click the microphone button**
3. **You should see**: Browser permission prompt appears
4. **Click "Allow"** in the prompt
5. **Status should change** to "🔴 Recording..."

## If Still Not Working

### Check Browser Console (F12):
Look for these messages:
- "Microphone access granted" ✓ Good
- "NotAllowedError" ✗ Permission still denied
- "NotFoundError" ✗ No microphone found

### Try Different Browser:
- **Chrome**: Usually works best
- **Firefox**: Also supports microphone
- **Edge**: Should work too

### Check System Microphone:
1. **Test in another app** (like Zoom, Discord)
2. **Check system settings** - make sure microphone isn't muted
3. **Try a different microphone** if available

## Common Issues

### "Permission denied" but I clicked Allow:
- **Clear browser cache** (Ctrl+Shift+Delete)
- **Restart browser**
- **Try incognito mode** (Ctrl+Shift+N)

### No permission prompt appears:
- **Check URL**: Must be `http://localhost:8000/ui/index.html` (not 0.0.0.0)
- **Check browser**: Some browsers block permissions on non-HTTPS sites
- **Check extensions**: Some extensions block microphone access

### Microphone works in other apps but not here:
- **Check browser permissions** (see Method 1 above)
- **Check if microphone is being used** by another app
- **Restart browser**
